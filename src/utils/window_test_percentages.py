import json

def window_test_percentages(windows_train, windows_test, title):
    train_dict = {}
    test_dict = {}
    for i, window_set in enumerate([windows_train, windows_test]):
        info_about_windows = train_dict if i == 0 else test_dict
        for window in window_set:
            activity_key = f"act_{window.activity}"
            if activity_key not in info_about_windows:
                info_about_windows[activity_key] = 1
            else:
                info_about_windows[activity_key] += 1
            
            subject_key = f"subj_{window.subject}"
            if subject_key not in info_about_windows:
                info_about_windows[subject_key] = 1
            else:
                info_about_windows[subject_key] += 1
    
    keys = set(list(train_dict.keys()) + list(test_dict.keys()))
    output_dict = {}
    print(f"{title} window_test_percentages:")
    for key in keys:
        if key not in train_dict:
            output_dict[key] = 1
        elif key not in test_dict:
            output_dict[key] = 0
        else:
            output_dict[key] = test_dict[key] / (train_dict[key] + test_dict[key])
    # sort dict
    output_dict = {k: v for k, v in sorted(output_dict.items(), key=lambda item: item[0], reverse=False)}
    print(json.dumps(output_dict, indent=4))
    

        
        
