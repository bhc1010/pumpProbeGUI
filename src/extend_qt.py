import numpy as np
import matplotlib.pyplot as plt

from src.colors import Color

plt.rcParams['toolbar'] = 'toolmanager'

from matplotlib.backend_tools import ToolBase
from PyQt5 import QtWidgets, QtCore

class QDataTableRow():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class QDataTable(QtWidgets.QTableWidget):
    def __init__(self, parent: QtWidgets.QWidget = None, read_only: bool = True):
        super().__init__(parent)
        self.data = list()
        self.setSelectionBehavior(QtWidgets.QTableView.SelectionBehavior.SelectRows)
        if read_only:
            self.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)

    def add_item(self, row: QDataTableRow, data=None):
        # if no optional data is given, then let new data be QDataTableRow dict 
        if data == None:
            data = row.__dict__
        # add new data to table's data array
        self.data.append(data)
        # add QDataTableRow object to table
        row_idx = len(self.data) - 1
        if row_idx + 1 > self.rowCount():
            self.insertRow(row_idx)
        for col_idx, key in enumerate(row.__dict__.keys()):
            self.setItem(row_idx, col_idx, QtWidgets.QTableWidgetItem(row.__dict__[key]))

class FlipData(ToolBase):
    def trigger(self, *args, **kwargs):
        ax = self.figure.axes[0]
        for line in self.figure.axes[0].lines[:-2]:
            voltage = np.array(line.get_data()[1])
            line.set_ydata(-voltage)
            
        ax.relim()
        ax.autoscale_view()

class AutocorLines(ToolBase):
    def trigger(self, *args, **kwargs):
        ax = self.figure.axes[0]
        vlines = ax.lines[-2:]
        vis_value = vlines[0].get_visible()
        for line in vlines:
            line.set_visible(not vis_value)
        
class QPlotter(QtCore.QObject):
    """
    """
    # Plotter signals are called inside pyppspec.pumpprobe
    _plot = QtCore.pyqtSignal(list)
    _new_line = QtCore.pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.xdata = list()
        self.ydata = list()
        self.lines = list()
        self.colors = Color.PASTELS(order='random')

    def mk_figure(self, info: list):
        self.clr_data()
        self.clr_lines()
        self.fig = plt.figure(figsize=(8,5))

        procedure_info, x_axis = info
        ax = self.fig.add_subplot(111)
        plt.title("Pump-probe Spectroscopy")
        plt.xlabel(x_axis, fontsize=12)
        plt.ylabel(r"Voltage (V)", fontsize=12)
        plt.grid(True)
        plt.subplots_adjust(right=0.725)

        info_display = ""
        for line in procedure_info:
            for i, text in enumerate(line):
                info_display = info_display + text
                if i == 0:
                    if '[' not in text:
                        if '-' not in text:
                            info_display += ' : '
            info_display += '\n'
        plt.text(1.05, 0.15, info_display, transform=ax.transAxes)
        
    def add_tools(self):
        # Add custom tools to figure
        self.fig.canvas.manager.toolmanager.add_tool('Flip Data', FlipData)
        self.fig.canvas.manager.toolbar.add_tool('Flip Data', 'custom')
        
        self.fig.canvas.manager.toolmanager.add_tool('Show/hide autocorrelation lines', AutocorLines)
        self.fig.canvas.manager.toolbar.add_tool('Show/hide autocorrelation lines', 'custom')

    def add_average_line(self):
        ax = self.fig.axes[0]
        self.average = ax.plot([0,0], c='black', label='Average', zorder=9999)[0]
        plt.legend()
        
    def add_line(self):
        self.clr_data()
        ax = plt.gca()
        line = ax.plot([0,0], c=next(self.colors).RGB())[0]
        self.lines.append(line)

    def set_line(self, line_params: dict):
        ax = self.fig.axes[-1]
        keys = line_params.keys()
        self.lines[-1].set(**{i:line_params[i] for i in keys if i not in ['data', 'error', 'fillcolor', 'fc']})
                
        if 'fillcolor' in keys or 'fc' in keys:
            try:
                fill_color = line_params['fillcolor']
            except:
                fill_color = line_params['fc']
            
        if 'data' in keys:
            dt, voltage = line_params['data']
            self.lines[-1].set_data(dt, voltage)
            if 'error' in keys:
                error = line_params['error']
                ax.fill_between(dt, voltage - error, voltage + error, color=fill_color, alpha=0.5)

    def update_figure(self, data:list = None):
        if data:
            self.add_data(data[0], data[1])
        
        # update current plot
        self.lines[-1].set_data(self.xdata, self.ydata)
        
        # update average plot
        if 'average' in self.__dict__.keys():
            cur_len = len(self.lines[-1].get_ydata())
            ydata = []
            for line in self.lines:
                ydata.append(line.get_ydata()[:cur_len])
            self.average.set_data(self.xdata, np.average(ydata, axis=0))            
        
        # rescale
        ax = plt.gca()
        ax.relim()
        ax.autoscale_view()
        
    def zero_line(self, zero: float):
        plt.axvline(zero, color = 'r', linestyle='--', visible=False)

    def add_data(self, x:float, y:float):
        self.xdata.append(x)
        self.ydata.append(y)
        
    def clr_data(self):
        self.xdata = list()
        self.ydata = list()
        
    def clr_lines(self):
        self.lines = list()