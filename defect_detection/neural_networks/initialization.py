def setup_neural_network(configuration, weights):
    '''
    :param configuration: YOLO configuration reflecting the network's inner parameters
    :param weights: YOLO weights
    :return: neural network initialized
    '''
    import cv2
    
    NN = cv2.dnn.readNetFromDarknet(configuration, weights)
    NN.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    NN.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    return NN

def load_files():
    '''
    Loads and returns necessary files: classes, configurations, weights
    :return: 
    RELATIVE PATHS INSIDE THE PROGRAM REQUIRED!
    '''
    classes_txt = r"C:\Users\Evgenii\Desktop\Python_Programming\Python_Projects\defect_detection\defect_detection\weights_configs\obj.names.txt"
    with open(classes_txt, "rt") as f:
        classes = f.read().rstrip("\n").split("\n")
    
    configuration_2_classes = r"C:\Users\Evgenii\Desktop\Python_Programming\Python_Projects\defect_detection\defect_detection\weights_configs\yolo-obj.cfg"
    configuration_3_classes = r"TBC"
    block_1_weights = r"D:\Desktop\Reserve_NNs\weights\Try3_Poles_MetalConcrete_Detection\yolo-obj_best.weights"
    block_2_weights_metal = r"D:\Desktop\Reserve_NNs\weights\Try4_Components_InsulatorsDumpers\yolo-obj_final.weights"
    block_2_weights_concrete = r"TBC"
    
    return classes, configuration_2_classes, configuration_3_classes, block_1_weights, block_2_weights_metal, block_2_weights_concrete
