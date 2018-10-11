from PyQt5.QtWidgets import QWidget, QHBoxLayout, \
    QLabel, QSpinBox

class GroupSpinBox(QWidget):
    """Группа спинбоксов с лейблами"""

    def __init__(self, spinboxes, parent=None):
        """
        Варианты содержимого spinboxes:
        list((QLabel, QSpinBox)) | list(QLabel) | list(QSpinBox)
        """
        QWidget.__init__(self, parent)
        self.initUI(spinboxes)

    def initUI(self, initlist):
        lo = QHBoxLayout()
        self.spinboxes = []

        for i, value in enumerate(initlist):

            if isinstance(value, str):
                title = QLabel(value)
                spinbox = QSpinBox()
            elif isinstance(value, QSpinBox):
                title = QLabel('')
                spinbox = value
            elif isinstance(value, tuple):
                title, spinbox = value
                if isinstance(title, str):
                    title = QLabel(title)
                elif not isinstance(title, QLabel):
                    raise NotImplementedError

            lo.addWidget(title)
            lo.addWidget(spinbox)
            self.spinboxes.append(spinbox)

        lo.addStretch()
        lo.setContentsMargins(4, 2, 4, 2)  # l t r b
        self.setLayout(lo)

    def value(self):
        value = tuple(sb.value() for sb in self.spinboxes)
        return value

    def setValue(self, value):
        for v, sb in zip(value, self.spinboxes):
            sb.setValue(v)

    def getSpinBoxAt(self, index):
        spinbox = self.spinboxes[index] if index < len(self.spinboxes) \
            else None
        return spinbox

    def getValueAt(self, index):
        spinbox = self.getSpinBoxAt(index)
        return spinbox.value() if spinbox else None

    def setValueAt(self, index, value):
        spinbox = self.getSpinBoxAt(index)
        spinbox.setValue(value)

    def setMinimumAt(self, index, minimum):
        spinbox = self.getSpinBoxAt(index)
        spinbox.setMinimum(minimum)

    def setMaximumAt(self, index, maximum):
        spinbox = self.getSpinBoxAt(index)
        spinbox.setMaximum(maximum)

    def setRangeAt(self, index, minimum, maximum):
        spinbox = self.getSpinBoxAt(index)
        spinbox.setRange(minimum, maximum)

    def setMinimum(self, minimum):
        for spinbox, m in zip(self.spinboxes, minimum):
            spinbox.setMinimum(m)

    def setMaximum(self, maximum):
        for spinbox, M in zip(self.spinboxes, maximum):
            spinbox.setMaximum(M)

    def setRange(self, ranges):
        for spinbox, (m, M) in zip(self.spinboxes, ranges):
            spinbox.setRange(m, M)

    def minimum(self):
        minimum = [spinbox.minimum() for spinbox in self.spinboxes]
        return minimum

    def maximum(self):
        maximum = [spinbox.maximum() for spinbox in self.spinboxes]
        return maximum

    def clear(self):
        minimum = self.minimum()
        self.setValue(minimum)