from .diagramaxis import DiagramAxis

class SpecAxis(DiagramAxis):
    def __init__(self, unit, limits=None, orig=None):
        super().__init__(limits=limits, orig=orig)

        if not isinstance(orig, SpecAxis):
            self.__unit = unit
        else:
            self.__unit = unit or orig.__unit

        if self.__unit == 'nm':
            self.label = r'$\lambda$ [nm]'
        elif self.__unit == 'AA':
            self.label = r'$\lambda$ [$\AA$]'
        else:
            raise NotImplementedError()

        self._validate()

    def _validate(self):
        pass

    def __get_unit(self):
        return self.__unit

    unit = property(__get_unit)