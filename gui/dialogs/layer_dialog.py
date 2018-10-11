from PyQt5.QtWidgets import QDialog, QWidget, QGroupBox, \
    QRadioButton, QComboBox, QSpinBox, \
    QFormLayout, QVBoxLayout, QDialogButtonBox

from gui.group_spinbox import GroupSpinBox


class LayerParams(QWidget):
    '''
        border_mode: str, int or tuple of three int
        Either of the following:

        ``'valid'``: применить фильтр везде, где он полностью перекрывает
            input. Генерирует output формы: input shape - filter shape + 1
        ``'full'``: применить фильтр везде, где он частично перекрывает input.
            Генерирует output формы: input shape + filter shape - 1
        ``'half'``: pad input with a symmetric border of ``filter // 2``,
            затем выполнить valid свертку. Для фильтров с нечетным
            числом slices, rows и columns, это приводит к тому, что форма output
            совпадает с формой input.
        ``int``: pad input with a symmetric border of zeros of the given
            width, then perform a valid convolution.
        ``(int1, int2, int3)``
            pad input with a symmetric border of ``int1``, ``int2`` and
            ``int3`` columns, then perform a valid convolution.

        subsample: tuple of len 3
            Factor by which to subsample the output.
            Also called strides elsewhere.

        filter_flip: bool
            If ``True``, will flip the filter x, y and z dimensions before
            sliding them over the input. This operation is normally
            referred to as a convolution, and this is the default. If
            ``False``, the filters are not flipped and the operation is
            referred to as a cross-correlation.

        filter_dilation: tuple of len 3
            Factor by which to subsample (stride) the input.
            Also called dilation elsewhere.
    '''

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.initUI()

    def initUI(self):

        filter_group = QGroupBox('filter')
        filter = GroupSpinBox(['count', 'size'])
        filter.setRange([(1, 500), (1, 100)])
        filter_group.setLayout(filter.layout())
        self.filter = filter

        predefined = QRadioButton('predefined')
        predefined.setChecked(True)
        self.predefined = predefined
        for_all = QRadioButton('border for all dimensions')
        self.for_all = for_all
        for_each = QRadioButton('border for each dimension')
        self.for_each = for_all

        # predefined
        border_mode_str = QComboBox()
        border_mode_str.addItems([
            'valid', 'full', 'half'
        ])
        border_mode_str.setCurrentIndex(0)
        self.border_mode_str = border_mode_str
        # border_mode_str.setEnabled(False)

        # symmetric border for all dimensions
        border_mode_int = QSpinBox()
        border_mode_int.setEnabled(False)
        self.border_mode_int = border_mode_int

        # symmetric border for each dimension
        border_mode_tuple = GroupSpinBox(['D', 'H', 'W'])
        border_mode_tuple.setEnabled(False)
        self.border_mode_tuple = border_mode_tuple

        glo = QFormLayout()
        glo.addRow(predefined, border_mode_str)
        glo.addRow(for_all, border_mode_int)
        glo.addRow(for_each, border_mode_tuple)

        border_mode_group = QGroupBox('border mode')
        border_mode_group.setLayout(glo)

        predefined.toggled[bool].connect(
            border_mode_str.setEnabled
        )

        for_all.toggled[bool].connect(
            border_mode_int.setEnabled
        )

        for_each.toggled[bool].connect(
            border_mode_tuple.setEnabled
        )

        subsample_group = QGroupBox('subsample')
        subsample = GroupSpinBox(['D', 'H', 'W'])
        subsample.setValue((1, 1, 1))
        subsample_group.setLayout(subsample.layout())
        self.subsample = subsample

        filter_dilation_group = QGroupBox('filter dilation')
        filter_dilation = GroupSpinBox(['D', 'H', 'W'])
        filter_dilation.setValue((1, 1, 1))
        filter_dilation_group.setLayout(filter_dilation.layout())
        self.filter_dilation = filter_dilation

        '''filter_flip = QCheckBox('filter flip')
        filter_flip.setLayoutDirection(Qt.RightToLeft)
        filter_flip.setChecked(True)
        self.filter_flip = filter_flip'''

        lo = QVBoxLayout()

        lo.addWidget(filter_group)

        lo.addWidget(border_mode_group)

        lo.addWidget(subsample_group)

        lo.addWidget(filter_dilation_group)

        # lo.addWidget(filter_flip)

        lo.addStretch()

        self.setLayout(lo)

    def border_mode(self):

        if self.border_mode_str.isEnabled():
            border_mode = self.border_mode_str.currentText()
        elif self.border_mode_int.isEnabled():
            border_mode = self.border_mode_int.value()
        else:
            border_mode = self.border_mode_tuple.value()
        return border_mode

    def value(self):
        value = {
            'filter_shape': self.filter.value(),
            'border_mode': self.border_mode(),
            'subsample': self.subsample.value(),
            # 'filter_flip': self.filter_flip.isChecked(),
            'filter_dilation': self.filter_dilation.value()
        }
        return value

    def setValue(self, value):

        filter_value = value.get('filter_shape', (1, 1))
        self.filter.setValue(filter_value)

        border_value = value.get('border_mode', 'valid')
        if isinstance(border_value, str):
            self.predefined.toggle()
            self.border_mode_str.setCurrentText(border_value)
        elif isinstance(border_value, int):
            self.for_all.toggle()
            self.border_mode_int.setValue(border_value)
        else:
            self.for_each.toggle()
            self.border_mode_tuple.setValue(border_value)

        subsample = value.get('subsample', (1, 1, 1))
        self.subsample.setValue(subsample)

        # filter_flip = value.get('filter_flip', True)
        # self.filter_flip.setChecked(filter_flip)

        dilation = value.get('dilation', (1, 1, 1))
        self.filter_dilation.setValue(dilation)

    def reset(self):
        self.filter.clear()
        self.border_mode_str.setCurrentIndex(0)
        self.border_mode_int.clear()
        self.border_mode_tuple.clear()
        self.predefined.toggle()
        self.subsample.clear()
        # self.filter_flip.setChecked(True)
        self.filter_dilation.clear()


class LayerDialog(QDialog):
    """
    Диалог для добавления слоя свертки в автокодировщик
    """

    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.initUI()

    def initUI(self):
        params = LayerParams()
        self.params = params

        buttons = QDialogButtonBox(
            QDialogButtonBox.Reset
            | QDialogButtonBox.Ok
            | QDialogButtonBox.Cancel
        )
        buttons.button(QDialogButtonBox.Reset).clicked.connect(self.reset)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        vlo = QVBoxLayout()
        vlo.addWidget(params)
        vlo.addWidget(buttons)

        self.setLayout(vlo)

    def reset(self):
        self.params.reset()

    def value(self):
        return self.params.value()

    def setValue(self, value):
        self.params.setValue(value)
