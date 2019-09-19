from visdom import Visdom
from datetime import datetime
import numpy as np

class VisTorch:

    DEFAULT_HOSTNAME = "127.0.0.1"
    DEFAULT_PORT = 8097
    VIS_CON = None

    def __init__(self, env_name = None):

        if env_name is None:
            env_name = str(datetime.now().strftime("%m%d%H%M%S"))
        self.env_name = env_name
        self.loss_window = None

    def close(self):
        self.VIS_CON.close()

    def __vis_initializer(self):

        if self.VIS_CON is None:
            self.VIS_CON = Visdom(server=self.DEFAULT_HOSTNAME, port=self.DEFAULT_PORT)

        assert self.VIS_CON.check_connection(timeout_seconds=3), 'No connection could be formed quickly'

    def plot_loss(self, epoch, *losses, loss_type='Loss', ytickmin = None, ytickmax = None):

        self.__vis_initializer()
        legend = ['Training', 'Evaluation', 'Training_1']
        linecolors = np.array([[0, 191, 255], [255, 10, 0], [255, 0, 255]])
        self.loss_window = self.VIS_CON.line(Y= np.column_stack(losses),
                                             X = np.column_stack([epoch]*len(losses)),
                                             win=self.loss_window,
                                             update='append' if self.loss_window else None,
                                             opts = {
                                                 'xlabel': 'Epoch',
                                                 'ylabel': loss_type,
                                                 'ytickmin': ytickmin,
                                                 'ytickmax': ytickmax,
                                                 'title': 'Learning curve',
                                                 'showlegend': True,
                                                 'linecolor': linecolors[:len(losses)],
                                                 'legend': legend[:len(losses)]
                                             })

