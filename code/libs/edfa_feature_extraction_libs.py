from .edfaBasicLib import *
'''
## split to sub-dataset for more easy combinations ....
### train
#### fix-baseline (without goalpost)
#### 80% random, extraLow, extraRandom
### test
#### goalpost without first 
#### 20% random
### augment
#### non-repeated fix without goalpost dataset
#### non-repeated goalpost
'''

def featureExtraction_ML(data_characterization, extractionType, channelType, featureType, channelNum=95, calculateRipple=False, newCalibration=True, calculateAugment=False):
    if featureType not in ["train","test","test_ground_truth"]:
        raise Exception("feature type not accurate.")
    ExtractedFeature = pd.DataFrame()
    for metadata in data_characterization:

        # different feature extraction from different measurements
        if extractionType == "preamp":

            repeat_index_start = 1
            # target gain & wss channels
            gain_target = metadata["roadm_dut_preamp_info"]["target_gain"]
            tilt_target = metadata["roadm_dut_preamp_info"]["target_gain_tilt"]
            wss_channels = metadata['roadm_flatten_wss_active_channel_index']
            # PD reading
            edfa_input_power_total = metadata["roadm_dut_preamp_info"]["input_power"]
            edfa_output_power_total= metadata["roadm_dut_preamp_info"]["output_power"]
            # spectra readings
            EDFA_input_spectra  = np.array(list(metadata["roadm_dut_preamp_input_power_spectra"].values()))
            if featureType == "train" or featureType == "test_ground_truth":
                EDFA_output_spectra = np.array(list(metadata["roadm_dut_wss_input_power_spectra"].values()))
                # print(EDFA_output_spectra)
                # print("roadm_dut_wss_input_power_spectra")
        
        elif extractionType == "booster":

            repeat_index_start = 0
            # target gain & wss channels
            gain_target = metadata["roadm_dut_edfa_info"]["target_gain"]
            tilt_target = metadata["roadm_dut_edfa_info"]["target_gain_tilt"]
            wss_channels = metadata["roadm_dut_wss_active_channel_index"]
            # PD reading
            edfa_input_power_total = metadata["roadm_dut_edfa_info"]["input_power"]
            edfa_output_power_total= metadata["roadm_dut_edfa_info"]["output_power"]
            # spectra readings
            EDFA_input_spectra = np.array(list(metadata["roadm_dut_wss_output_power_spectra"].values()))
            if featureType == "train" or featureType == "test_ground_truth":
                EDFA_output_spectra = np.array(list(metadata["roadm_dut_booster_output"].values()))
                # print(EDFA_output_spectra)
                # print("roadm_dut_booster_output")

        else:
            print(extractionType+" has not implemented!")
            exit(-1)

        # check if the spectra is corre
        
        # take training data in fix channel loading
        if channelType == "fix" and featureType == "train":
            # ignore all the goalpost since they are in test set
            if metadata["open_channel_type"] == "goalpost_channel_balanced_freq_low_medium":
                break
        
        # take test data for goalpost
        if channelType == "fix" and (featureType == "test" or featureType == "test_ground_truth"):
            # only take goalpost
            if "goalpost" not in metadata["open_channel_type"]:
                continue
            else: # ignore first repeat in the test set
                if "repeat_index" in metadata.keys():
                    repeat_index = metadata["repeat_index"]
                    # since booster start with 0 but preamp start with 1 ...
                    if repeat_index == repeat_index_start:
                        continue

        # gain profile
        acutal_gain_spectra = []
        if featureType == "train" or featureType == "test_ground_truth": 
            acutal_gain_spectra = EDFA_output_spectra - EDFA_input_spectra
            if calculateRipple:
                acutal_gain_spectra -= gain_target
        # calculate one hot DUT WSS open channel
        DUT_WSS_activated_channels = [0]*channelNum
        for indx in wss_channels:
            DUT_WSS_activated_channels[indx-1] = 1

        # write the PD power info
        metaResult = {}
        metaResult['target_gain'] = gain_target
        metaResult['target_gain_tilt'] = tilt_target
        metaResult['EDFA_input_power_total'] = edfa_input_power_total
        metaResult['EDFA_output_power_total'] = edfa_output_power_total
        
        # write the spectra info
        for i in range(channelNum):
            post_indx = str(i).zfill(2)
            metaResult['EDFA_input_spectra_'+post_indx] = round(EDFA_input_spectra[i], 1)
            metaResult['DUT_WSS_activated_channel_index_'+post_indx] = DUT_WSS_activated_channels[i]
            if featureType == "train" or featureType == "test_ground_truth":
                metaResult['calculated_gain_spectra_'+post_indx] = round(acutal_gain_spectra[i], 1)
        
        ExtractedFeature = pd.concat([ExtractedFeature,pd.DataFrame.from_dict([metaResult])],ignore_index=True)

    return ExtractedFeature
