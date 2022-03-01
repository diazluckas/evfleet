from pathlib import Path
import os
import pandas as pd
import xml.etree.ElementTree as ET

def get_root(filepath: Path):
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
    except ET.ParseError:
        raise ET.ParseError(f"Not well-formed at file:\n {filepath}")
    
    return root

def get_data_from_root(root= None):
    for el in root:
        data = el.attrib
    return a

def save_data (root_one, root_two, root_three):
    for f, h, m in root_one, root_two, root_three:
        print(f, h ,m)
        breakpoint()
    

def save_to_csv(source_folder: Path, save_folder: Path):
    for file in source_folder.iterdir():
        if file.stem == 'fleet_end':
            fleet_filepath = file
        elif file.stem == 'history':
            history_filepath = file
        elif file.stem == 'measurements':
            measurements_filepath = file
    
    root_fleet = get_root(fleet_filepath)
    root_history = get_root(history_filepath)
    root_measurements = get_root(measurements_filepath)

    save_data(root_fleet, root_history, root_measurements)
    breakpoint()

    fleet_dict = get_data_from_root(root_fleet)
    history_dict = get_data_from_root(root_history)
    measurements_dict = get_data_from_root(root_measurements)

    fleet_data = pd.DataFrame(data= fleet_dict, index=[0])
    history_data = pd.DataFrame(data= history_dict, index=[0])
    measurements_data = pd.DataFrame(data= measurements_dict, index=[0])

    day_data = pd.concat([fleet_data, history_data, measurements_data], axis=1)
    day_data.reset_index(inplace=True, drop=True)

    
    if not os.path.isfile(Path(save_folder,'data.csv')):
        day_data.to_csv(Path(save_folder,'data.csv'))

    else:
        day_data.to_csv(Path(save_folder,'csvAux.csv'))
        day_data = pd.read_csv(Path(save_folder,'csvAux.csv'))
        day_data.drop(day_data.filter(regex='Unname'), axis=1, inplace=True)
        day_data.reset_index(inplace=True, drop=True)

        previous_day_data = pd.read_csv(Path(save_folder,'data.csv'))
        previous_day_data.drop(previous_day_data.filter(regex='Unname'), axis=1, inplace=True)
        previous_day_data.reset_index(inplace=True, drop=True)

        full_data = pd.concat([previous_day_data, day_data], axis=0, ignore_index=True)
        full_data.to_csv(Path(save_folder,'data.csv'))

day_folder = Path('data/test2/simulations/simulations_OpenLoop_degradation/policy_20_95/11-21-14_01-03-22_aramis_energy/day_data')
save_to_csv(day_folder, day_folder.parent)


#source = Path('data/test/simulations/simulations_OpenLoop_degradation/policy_20_95/18-07-23_28-02-22_aramis_energy')