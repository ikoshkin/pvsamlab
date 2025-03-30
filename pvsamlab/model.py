"""
PV Simulation module.

Defines classes to run pv models
"""
import numpy as np

import copy

ssc = PySSC()


class SSCmodel:
    def __init__(self, parameters, ssc_module_name='pvsamv1'):
        self._ssc_data = None
        self.ssc_module_name = ssc_module_name
        self.parameters = parameters

    def run(self, **params_kws):
        """ Run SAM SDK engine """

        if params_kws is not None:
            self.parameters.update(params_kws)

        self._ssc_data = ssc.data_create()

        module = ssc.module_create(self.ssc_module_name)

        ssc.data_load_from_dict(self._ssc_data, self.parameters)
        ssc.execute_ssc_module(module, self._ssc_data, self.ssc_module_name)

        ssc.module_free(module)
        return self

    def predict(self, ssc_output_key):
        return np.asarray(ssc.get_ssc_value(self._ssc_data, ssc_output_key))

    def run_predict(self, ssc_output_key, **params_kws):
        """NOTE: Be mindful with passing parameters updates - this will only 
        update a single key-value. (It should be safe to pass solar resource files etc.)
        Consider instantiating a new object for system configuration options.
        """

        self.run(**params_kws)
        return self.predict(ssc_output_key)

    retrieve = predict
