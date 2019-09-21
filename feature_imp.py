from sklearn import metrics
import numpy as np
import torch
import importlib
import sys
import OT_crispr_attn
import torch.nn.functional as F


config_path = ".".join(["models", sys.argv[1]]) + "." if len(sys.argv) >= 2 else ""
config = importlib.import_module(config_path + "config")

class Ranking(object):
    def __init__(self, names):
        self.names = names

    def _normalize(self, impt, fea_num):
        impt = impt / sum(impt)
        impt = list(zip(impt, self.names, range(fea_num)))
        impt.sort(key=lambda x: -x[0])
        return impt


class InputPerturbationRank(Ranking):
    def __init__(self, names):
        super(InputPerturbationRank, self).__init__(names)

    def _raw_rank(self, rep, y, network, x):

        fea_num = 0
        for fea in x:
            fea_num += int(fea.shape[1])
        impt = np.zeros(fea_num)

        fea_index = 0
        for fea_dfs in x:
            for i in range(fea_dfs.shape[1]):
                hold = np.array(fea_dfs[:, i])
                for j in range(rep):
                    np.random.shuffle(fea_dfs[:, i])

                    # Handle both TensorFlow and SK-Learn models.
                    if 'tensorflow' in str(type(network)).lower():
                        pred = list(network.predict(x))
                    else:
                        pred = network.predict_proba(fea_dfs)

                    #rmse = metrics.mean_squared_error(y, pred)
                    #spearman_correlation = spearmanr(y, pred)[0]
                    rmse = F.nll_loss(torch.FloatTensor(pred), torch.FloatTensor(y).long())
                    impt[fea_index] += (rmse - impt[fea_index]) / (j + 1)

                fea_index += 1
                fea_dfs[:, i] = hold

        return impt, fea_num

    def _torch_raw_rank(self, rep, y, network, x, batch_size = None, classifier = False):

        fea_num = 0
        for fea in x:
            fea_num += int(fea.shape[1])
        impt = np.zeros(fea_num)

        fea_index = 0
        with torch.set_grad_enabled(False):
            for fea_dfs in x:
                for i in range(fea_dfs.shape[1]):
                    hold = np.array(fea_dfs[:, i])
                    for j in range(rep):
                        np.random.shuffle(fea_dfs[:, i])
                        n_total = fea_dfs.shape[0]
                        if batch_size is None:
                            batch_size = n_total
                        preds_ls = []
                        for k in range((n_total+batch_size-1)//batch_size):
                            start = k*batch_size
                            end = min((k+1)*batch_size, n_total)
                            local_batch = torch.FloatTensor(fea_dfs[start: end, :]).to(OT_crispr_attn.device2)
                            preds = network(local_batch)
                            prediction_on_cpu = preds.cpu().numpy()
                            preds_ls.append(prediction_on_cpu)
                        pred = np.concatenate(tuple(preds_ls))

                        if not classifier:
                            rmse = metrics.mean_squared_error(y.reshape(-1), pred.reshape(-1))
                        else:
                            rmse = F.nll_loss(torch.FloatTensor(pred), torch.FloatTensor(y.reshape(-1)).long())
                        #spearman_correlation = spearmanr(y, pred)[0]
                        impt[fea_index] += (rmse - impt[fea_index]) / (j + 1)

                    fea_index += 1
                    fea_dfs[:, i] = torch.FloatTensor(hold)

        return impt, fea_num

    def rank(self, rep, y, network, x, torch = False, classifier=False):
        if torch:
            impt, fea_num = self._torch_raw_rank(rep, y, network, x, batch_size=config.batch_size, classifier=classifier)
        else:
            impt, fea_num = self._raw_rank(rep, y, network, x)
        return self._normalize(impt, fea_num)

