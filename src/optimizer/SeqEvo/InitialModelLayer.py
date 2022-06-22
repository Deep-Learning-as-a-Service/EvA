from model_representation.ParametrizedLayer.PConv2DLayer import Conv2DFiltersParam, Conv2DKernelSizeParam, Conv2DStridesParam, PConv2DLayer
from model_representation.ParametrizedLayer.PDenseLayer import DenseUnitsParam, PDenseLayer
from model_representation.ParametrizedLayer.PLstmLayer import LstmUnitsParam, PLstmLayer, LstmDropoutParam

class InitialModelLayer():
    
    @staticmethod
    def get_all_models():     
        model_list = []
        public_method_names = [method for method in dir(InitialModelLayer()) if callable(getattr(InitialModelLayer(), method)) if not (method.startswith('_') or method.startswith('get_all_models'))]  
        for method in public_method_names:
            model_list.append(getattr(InitialModelLayer(), method)())
        return model_list
    
    @staticmethod
    def triple_2dconv_lstm_1():      
        first_kernel = Conv2DKernelSizeParam.create((2,2))
        first = PConv2DLayer(
            params=[
                Conv2DFiltersParam.create(64),
                first_kernel,
                Conv2DStridesParam.create(("100%","step1"), dependent_on_param=first_kernel)
            ]
        )
        second_kernel = Conv2DKernelSizeParam.create((2,2))
        second = PConv2DLayer(
            params=[
                Conv2DFiltersParam.create(128),
                second_kernel,
                Conv2DStridesParam.create(("100%","step1"), dependent_on_param=second_kernel)
            ]
        )
        third_kernel = Conv2DKernelSizeParam.create((2,2))
        third = PConv2DLayer(
            params=[
                Conv2DFiltersParam.create(256),
                third_kernel,
                Conv2DStridesParam.create(("100%","step1"), dependent_on_param=third_kernel)
            ]
        )
        fourth = PLstmLayer(params=[LstmUnitsParam.create(512), LstmDropoutParam.create(0.1)])  
        return [ 
                first, second, third, fourth
            ]
        
    @staticmethod
    def triple_2dconv_lstm_2():      
        first_kernel = Conv2DKernelSizeParam.create((3,3))
        first = PConv2DLayer(
            params=[
                Conv2DFiltersParam.create(32),
                first_kernel,
                Conv2DStridesParam.create(("100%","100%"), dependent_on_param=first_kernel)
            ]
        )
        second_kernel = Conv2DKernelSizeParam.create((3,3))
        second = PConv2DLayer(
            params=[
                Conv2DFiltersParam.create(64),
                second_kernel,
                Conv2DStridesParam.create(("100%","100%"), dependent_on_param=second_kernel)
            ]
        )
        third_kernel = Conv2DKernelSizeParam.create((3,3))
        third = PConv2DLayer(
            params=[
                Conv2DFiltersParam.create(128),
                third_kernel,
                Conv2DStridesParam.create(("100%","100%"), dependent_on_param=third_kernel)
            ]
        )
        fourth = PLstmLayer(params=[LstmUnitsParam.create(256), LstmDropoutParam.create(0.1)])  
        return [ 
                first, second, third, fourth
            ]
    
    @staticmethod
    def double_2dconv_dense_1():      
        first_kernel = Conv2DKernelSizeParam.create((3,3))
        first = PConv2DLayer(
            params=[
                Conv2DFiltersParam.create(32),
                first_kernel,
                Conv2DStridesParam.create(("100%","100%"), dependent_on_param=first_kernel)
            ]
        )
        second_kernel = Conv2DKernelSizeParam.create((3,3))
        second = PConv2DLayer(
            params=[
                Conv2DFiltersParam.create(64),
                second_kernel,
                Conv2DStridesParam.create(("100%","100%"), dependent_on_param=second_kernel)
            ]
        )
        
        third = PDenseLayer(params=[DenseUnitsParam.create(1024)])  
        return [ 
                first, second, third
            ]
    
    @staticmethod
    def double_2dconv_dense_2():      
        first_kernel = Conv2DKernelSizeParam.create((2,2))
        first = PConv2DLayer(
            params=[
                Conv2DFiltersParam.create(64),
                first_kernel,
                Conv2DStridesParam.create(("step1","step1"), dependent_on_param=first_kernel)
            ]
        )
        second_kernel = Conv2DKernelSizeParam.create((2,2))
        second = PConv2DLayer(
            params=[
                Conv2DFiltersParam.create(128),
                second_kernel,
                Conv2DStridesParam.create(("step1","step1"), dependent_on_param=second_kernel)
            ]
        )
        
        third = PDenseLayer(params=[DenseUnitsParam.create(1024)])  
        return [ 
                first, second, third
            ]
        
    @staticmethod
    def leander_deep_conv_1():      
        first_kernel = Conv2DKernelSizeParam.create((5,1))
        first = PConv2DLayer(
            params=[
                Conv2DFiltersParam.create(32),
                first_kernel,
                Conv2DStridesParam.create(("100%","100%"), dependent_on_param=first_kernel)
            ]
        )
        second_kernel = Conv2DKernelSizeParam.create((5,1))
        second = PConv2DLayer(
            params=[
                Conv2DFiltersParam.create(64),
                second_kernel,
                Conv2DStridesParam.create(("100%","100%"), dependent_on_param=second_kernel)
            ]
        )
        third = PLstmLayer(params=[LstmUnitsParam.create(32), LstmDropoutParam.create(0.1)])  
        return [ 
                first, second, third
            ]
        
    @staticmethod
    def leander_deep_conv_2():      
        first_kernel = Conv2DKernelSizeParam.create((10,1))
        first = PConv2DLayer(
            params=[
                Conv2DFiltersParam.create(128),
                first_kernel,
                Conv2DStridesParam.create(("100%","100%"), dependent_on_param=first_kernel)
            ]
        )
        second_kernel = Conv2DKernelSizeParam.create((3,1))
        second = PConv2DLayer(
            params=[
                Conv2DFiltersParam.create(256),
                second_kernel,
                Conv2DStridesParam.create(("100%","100%"), dependent_on_param=second_kernel)
            ]
        )
        third = PLstmLayer(params=[LstmUnitsParam.create(64), LstmDropoutParam.create(0.1)])  
        return [ 
                first, second, third
            ]
        
    @staticmethod
    def jens_1():      
        first_kernel = Conv2DKernelSizeParam.create((3,3))
        first = PConv2DLayer(
            params=[
                Conv2DFiltersParam.create(32),
                first_kernel,
                Conv2DStridesParam.create(("50%","step1"), dependent_on_param=first_kernel)
            ]
        )
        second_kernel = Conv2DKernelSizeParam.create((3,3))
        second = PConv2DLayer(
            params=[
                Conv2DFiltersParam.create(64),
                second_kernel,
                Conv2DStridesParam.create(("50%","step1"), dependent_on_param=second_kernel)
            ]
        )
        third_kernel = Conv2DKernelSizeParam.create((3,3))
        third = PConv2DLayer(
            params=[
                Conv2DFiltersParam.create(128),
                third_kernel,
                Conv2DStridesParam.create(("50%","step1"), dependent_on_param=third_kernel)
            ]
        )
        fourth = PDenseLayer(params=[DenseUnitsParam.create(1024)])
        return [ 
                first, second, third, fourth
            ]
        
    @staticmethod
    def conv_lstm_1():  
        first_kernel = Conv2DKernelSizeParam.create((10, 1))
        first = PConv2DLayer(
            params=[
                Conv2DFiltersParam.create(128),
                first_kernel,
                Conv2DStridesParam.create(("100%","100%"), dependent_on_param=first_kernel)
            ]
        )
        second = PDenseLayer(params=[DenseUnitsParam.create(2056)])
        return [ 
                first, second
            ]
    
    
    
    
    
    
    