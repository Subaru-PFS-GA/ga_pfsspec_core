from .diagramaxis import DiagramAxis

class FluxAxis(DiagramAxis):
    def __init__(self, unit, limits=None, orig=None):
        super().__init__(limits=limits, orig=orig)

        if not isinstance(orig, FluxAxis):
            self.__unit = unit
        else:
            self.__unit = unit or orig.__unit

        if self.__unit == 'nJy':
            self.label = r'$F_\nu$ [nJy]'
        elif self.__unit == 'cgs_A':
            self.label = r'$F_\lambda$ [erg s$^{-1}$ c$^{-2}$ $\AA^{-1}$]'
        elif self.__unit == 'ADU':
            self.label = r'$F$ [ADU]'
        else:
            raise NotImplementedError()

        self._validate()

    def _validate(self):
        pass

    def __get_unit(self):
        return self.__unit

    unit = property(__get_unit)