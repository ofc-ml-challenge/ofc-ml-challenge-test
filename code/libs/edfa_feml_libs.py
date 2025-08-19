# some comments
import statistics
from typing import DefaultDict
from .edfaExternalLibs import *

#####################################################################
####################  Feature extraction help function (for ML)
#####################################################################

def readCSVList(cvsFile):
    # simply read csv file and return the column "nameOfColumn"
    nameData_pd = pd.read_csv(cvsFile)
    return nameData_pd["Name"],nameData_pd["Channel"]

def logsumexp10(x):
    return 10*np.log10(np.sum(np.power(10,x/10)))

Lumentum_ocm_threshold = {
    "booster in" : (-35.9,3.8),
    "booster out" : (-14.9,6.8),
    "line in" : (-35.4,-3.0),
    "preamp out" : (-14.6,6.8),
    "mux in" : (-28.1,0.3)
}

def returnValidOCMListdata(dataset,key):
    returnData = []
    for data in dataset:
        if data < Lumentum_ocm_threshold[key][0]:
            continue# drop and do nothing
        elif data > Lumentum_ocm_threshold[key][1]:
            raise ValueError("one channel ("+str(data)+" dBm) above the threshold of "+key+"("+str(Lumentum_ocm_threshold[key][1])+")")
        else:
            returnData.append(data)
    return np.array(returnData)

def featureExtraction(data_characterization,extractionType,channelNum=95,calculateRipple=False,\
    newCalibration=True,calculateAugment = False):
    # input
    #   input power spectra (list, non-normalization)
    #   total input power (single value)
    #   target gain       (single value)
    #   (can extend with other settings)
    # output
    #   calculated gain spectra (list, non-normalization)
    
    ExtractedFeature = pd.DataFrame()
    for metadata in data_characterization:

        # different feature extraction from different measurements
        if extractionType == "preamp":

            repeat_index_start = 1
            # target gain & wss channels
            gain_target = metadata["roadm_dut_preamp_info"]["target_gain"]
            wss_channels = metadata['roadm_flatten_wss_active_channel_index']
            # PD reading
            edfa_input_power_total = metadata["roadm_dut_preamp_info"]["input_power"]
            edfa_output_power_total= metadata["roadm_dut_preamp_info"]["output_power"]
            # spectra readings
            EDFA_input_spectra  = np.array(list(metadata["roadm_dut_preamp_input_power_spectra"].values()))
            EDFA_output_spectra = np.array(list(metadata["roadm_dut_wss_input_power_spectra"].values()))
        
        elif extractionType == "booster":

            repeat_index_start = 0
            # target gain & wss channels
            gain_target = metadata["roadm_dut_edfa_info"]["target_gain"]
            wss_channels = metadata["roadm_dut_wss_active_channel_index"]
            # PD reading
            edfa_input_power_total = metadata["roadm_dut_edfa_info"]["input_power"]
            edfa_output_power_total= metadata["roadm_dut_edfa_info"]["output_power"]
            # spectra readings
            EDFA_input_spectra = np.array(list(metadata["roadm_dut_wss_output_power_spectra"].values()))
            EDFA_output_spectra = np.array(list(metadata["roadm_dut_booster_output"].values()))
        
        elif extractionType == "1span":
            
            repeat_index_start = 0 # default
            # target gain & wss channels
            gain_target = metadata["roadm_booster_preamp_info"]["target_gain"]
            wss_channels = metadata["roadm_booster_wss_active_channel_index"]
            # PD reading
            edfa_input_power_total = metadata["roadm_booster_preamp_info"]["input_power"]
            edfa_output_power_total= metadata["roadm_preamp_preamp_info"]["output_power"]
            # spectra readings
            EDFA_input_spectra = np.array(list(metadata["roadm_booster_wss_input_power_spectra"].values()))
            EDFA_output_spectra = np.array(list(metadata["roadm_preamp_wss_input_power_spectra"].values()))

        elif extractionType == "2span":

            repeat_index_start = 0 # default
            # target gain & wss channels
            gain_target = metadata["roadm_1_booster_info"]["target_gain"]
            wss_channels = metadata["open_channel_list"]
            # PD reading
            edfa_input_power_total = metadata["roadm_1_booster_info"]["input_power"]
            edfa_output_power_total= metadata["roadm_2_preamp_info"]["output_power"]
            # spectra readings
            EDFA_input_spectra = np.array(list(metadata["roadm_1_wss_out"].values()))
            EDFA_output_spectra = np.array(list(metadata["roadm_2_wss_in"].values()))

        else:
            print(extractionType+" has not implemented!")
            exit(-1)

        # skip repeated fix channel loading for augment data
        if "repeat_index" in metadata.keys():
            repeat_index = metadata["repeat_index"]
            # since booster start with 0 but preamp start with 1 ...
            if calculateAugment == True and repeat_index > repeat_index_start:
                continue

        # gain profile
        acutal_gain_spectra = []
        for i in range(channelNum):
            if calculateRipple:
                gain = EDFA_output_spectra[i] - EDFA_input_spectra[i] - gain_target # - delta1 - delta2
            else:
                gain = EDFA_output_spectra[i] - EDFA_input_spectra[i] # - delta1 - delta2
            acutal_gain_spectra.append(gain)
        # calculate one hot DUT WSS open channel
        DUT_WSS_activated_channels = [0]*channelNum
        for indx in wss_channels:
            DUT_WSS_activated_channels[indx-1] = 1

        # write the PD power info
        metaResult = {}
        # HeaderCSV = ['EDFA_input_spectra','EDFA_input_power_total','target_gain','calculated_gain_spectra']
        metaResult['target_gain'] = gain_target
        metaResult['EDFA_input_power_total'] = edfa_input_power_total
        metaResult['EDFA_output_power_total'] = edfa_output_power_total
        
        # write the spectra info
        for i in range(channelNum):
            post_indx = str(i).zfill(2)
            metaResult['EDFA_input_spectra_'+post_indx] = EDFA_input_spectra[i]
            metaResult['DUT_WSS_activated_channel_index_'+post_indx] = DUT_WSS_activated_channels[i]
            metaResult['calculated_gain_spectra_'+post_indx] = acutal_gain_spectra[i]
            metaResult['EDFA_output_spectra_'+post_indx] = EDFA_output_spectra[i]
        
        # ExtractedFeature = ExtractedFeature.append([metaResult],ignore_index=True)
        ExtractedFeature = pd.concat([ExtractedFeature,pd.DataFrame.from_dict([metaResult])],ignore_index=True)

    return ExtractedFeature

class edfaInfo():
    def __init__(self,dataset_folder,edfaType,gain,fileName):
        self.dataset_folder = dataset_folder
        self.edfaType = edfaType
        self.gain = gain
        self.fileName = fileName

def generateCMfile(edfaInfo,output_folder):
    results = {}
    folderList = ['fix', 'extraLow']
    for channelType in folderList:
        filePath = edfaInfo.dataset_folder+edfaInfo.edfaType+"/"+\
                        edfaInfo.gain+"/"+ channelType + "/"
        fileName_edfa = matchFile('*'+edfaInfo.fileName+'*.json', filePath)

        with open(fileName_edfa, "r") as read_file:
            data = json.load(read_file)

        # feature extraction
        if channelType == "fix":
            newFeature = CMfeatureExtraction(
                data["measurement_data"], edfaInfo.edfaType, whetherFullyLoading=True)
            results["wdm"] = newFeature
        elif channelType == 'extraLow':
            newFeature = CMfeatureExtraction(
                data["measurement_data"], edfaInfo.edfaType, whetherFullyLoading=False)
            results["single"] = newFeature
        else:
            print("haven't implemented yet!")
    output_path = output_folder+edfaInfo.gain+"/"+edfaInfo.fileName+".json"
    with open(output_path, 'w') as file_output:
        json.dump(results, file_output, indent=4)


def CMfeatureExtraction(data_characterization,edfaType,whetherFullyLoading=True,channelNum=95):
    
    if whetherFullyLoading:
        ExtractedFeature = []
        for metadata in data_characterization:

            if metadata["open_channel_type"] != "fully_loaded_channel_wdm":
                continue

            if edfaType == "preamp":
                EDFA_input_power_spectra = np.array(list(metadata["roadm_dut_preamp_input_power_spectra"].values()))
                EDFA_output_power_spectra = np.array(list(metadata["roadm_dut_wss_input_power_spectra"].values()))
            else:
                EDFA_input_power_spectra = np.array(list(metadata["roadm_dut_wss_output_power_spectra"].values()))
                EDFA_output_power_spectra = np.array(list(metadata["roadm_dut_booster_output"].values()))

            # gain profile
            acutal_gain_spectra = []
            for i in range(channelNum):
                gain = EDFA_output_power_spectra[i] - EDFA_input_power_spectra[i]
                acutal_gain_spectra.append(gain)
            ExtractedFeature.append(acutal_gain_spectra)

        # hand write average across repeats measurements across different attenuations
        result = []
        for i in range(channelNum):
            result.append(statistics.fmean([ExtractedFeature[j][i] for j in range(len(ExtractedFeature))]))

    else:
        ExtractedFeature = DefaultDict(list)
        for metadata in data_characterization:
            if metadata["open_channel_type"] != "single_channel":
                continue

            if edfaType == "preamp":
                EDFA_input_power_spectra = metadata["roadm_dut_preamp_input_power_spectra"]
                EDFA_output_power_spectra = metadata["roadm_dut_wss_input_power_spectra"]
                indx = str(metadata["roadm_flatten_wss_active_channel_index"][0])
            else:
                indx = str(metadata["roadm_dut_wss_active_channel_index"][0])
                EDFA_input_power_spectra = metadata["roadm_dut_wss_output_power_spectra"]
                EDFA_output_power_spectra = metadata["roadm_dut_booster_output"] 
            
            # gain profile
            gain = EDFA_output_power_spectra[indx] -EDFA_input_power_spectra[indx]
            ExtractedFeature[indx].append(gain)
            # hand write average across repeats measurements across different attenuations
        
        result = []
        for indx in range(channelNum):
            result.append(statistics.fmean(ExtractedFeature[str(indx+1)]))
    
    return result

def featureExtractionFromFile(fileName,whetherPreamp):
    with open(fileName, "r") as read_file:
        data = json.load(read_file)

    # feature extraction
    newFeature = featureExtraction(data["measurement_data"],whetherPreamp)

    return newFeature

def matchFile(pattern, foler):
    # match one file in the folder
    # example usage:
    # result = matchFile('*rdm1-co1*.json', '.../benchmark/extraRandom/')
    # result is the full path 
    for file in os.listdir(foler):
        if fnmatch.fnmatch(file, pattern):
            return os.path.join(foler, file)
