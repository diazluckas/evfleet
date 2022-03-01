import datetime
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Union, Tuple, Type

import numpy as np
from sklearn.neighbors import NearestNeighbors

import pandas as pd
import os

from res.dispatcher import Dispatcher
from res.optimizer.GATools import OnGA_HyperParameters as HyperParameters
from res.results import OnlineResults
from res.simulator import Simulator, SimulatorLV
from res.models import Fleet, Network

from res.stages import PreOperation


@dataclass
class OnlineParameters:
    source_folder: Path
    simulation_folder: Path
    optimize: bool
    soc_policy: Tuple[float, float]
    hyper_parameters: Union[HyperParameters, None]
    keep_times: int = 1
    sample_time: Union[float, int] = 300
    std_factor: float = 1.0
    start_earlier_by: float = 600.
    ev_type: Type[Fleet.EV.ElectricVehicle] = None
    fleet_type: Type[Fleet.Fleet] = None
    edge_type: Type[Network.Edge.DynamicEdge] = None
    network_type: Type[Network.Network] = None

    network_path: Path = None
    fleet_path: Path = None
    routes_path: Path = None
    eta_model: NearestNeighbors = None
    eta_table: np.ndarray = None

    def __post_init__(self):
        self.fleet_path = Path(self.source_folder, "instance.xml")
        self.network_path = Path(self.source_folder, "instance.xml")
        if not self.fleet_path.is_file():
            self.fleet_path = Path(self.source_folder, "fleet.xml")
            self.network_path = Path(self.source_folder, "network.xml")
        self.routes_path = Path(self.source_folder, "routes.xml")

    def __str__(self):
        s = ""
        skip = ["hyper_parameters", "eta_model", "eta_table"]
        for (key, val) in {x: y for x, y in self.__dict__.items() if x not in skip}:
            s += f"        {key}:  {val}\n"
        return s


def online_operation(main_folder: Path, source_folder: Path, optimize: bool = False, onGA_hp: HyperParameters = None,
                     repetitions: int = 5, hold_by: int = 0, sample_time: float = 300.,
                     std_factor: float = 1.0, start_earlier_by: float = 600,
                     soc_policy: Tuple[float, float] = (20., 95), display_gui: bool = False,
                     ev_type: Type[Fleet.EV.ElectricVehicle] = Fleet.EV.ElectricVehicle,
                     fleet_type: Type[Fleet.Fleet] = Fleet.Fleet,
                     edge_type: Type[Network.Edge.DynamicEdge] = Network.Edge.DynamicEdge,
                     network_type: Type[Network.Network] = Network.Network):
    simulation_type = 'closed_loop' if optimize else 'open_loop'
    simulations_folder = Path(main_folder, simulation_type)
    if display_gui:
        if not source_folder.is_dir():
            print("Directory is not valid: ", source_folder)
            return 0
        print("Will simulate results from:\n  ", source_folder)
        print("Simulation results will be saved to:\n  ", simulations_folder)
        input("Press ENTER to continue... (ctrl+Z to end process)")

    simulations_folder = Path()
    for i in range(repetitions):
        simulations_folder = simulate(main_folder, source_folder, optimize, onGA_hp, hold_by, sample_time, std_factor,
                                      start_earlier_by, soc_policy, display_gui, None, None, ev_type, fleet_type,
                                      edge_type, network_type)

    # Summarize results
    df_costs, df_constraints = OnlineResults.folder_data(simulations_folder.parent, source_folder)
    df_costs.to_csv(Path(simulations_folder.parent, 'costs.csv'))
    df_constraints.to_csv(Path(simulations_folder.parent, 'constraints.csv'))


def simulate(main_folder: Path, source_folder: Path = None, optimize: bool = False, onGA_hp: HyperParameters = None,
             hold_by: int = 0, sample_time: float = 300.0, std_factor: float = 1.0, start_earlier_by: float = 600,
             soc_policy: Tuple[float, float] = (20., 95), display_gui: bool = True,
             previous_simulation_folder: Path = None, simulation_name: str = None,
             ev_type: Type[Fleet.EV.ElectricVehicle] = Fleet.EV.ElectricVehicle,
             fleet_type: Type[Fleet.Fleet] = Fleet.Fleet,
             edge_type: Type[Network.Edge.DynamicEdge] = Network.Edge.DynamicEdge,
             network_type: Type[Network.Network] = Network.Network):
    if simulation_name is None:
        simulation_name = datetime.datetime.now().strftime("%H-%M-%S_%d-%m-%y")
    simulation_type = 'closed_loop' if optimize else 'open_loop'
    simulations_folder = Path(main_folder, simulation_type)
    simulation_folder = Path(simulations_folder, simulation_name)

    p = OnlineParameters(source_folder, simulation_folder, optimize, soc_policy, onGA_hp, hold_by, sample_time,
                         std_factor, start_earlier_by, ev_type, fleet_type, edge_type, network_type)
    if display_gui:
        text = f"""The simulation will run using the following parameters:
{p}
Simulation results will be saved to:
        {simulation_folder}
Press any key to continue..."""
        input(text)
    else:
        print(f"Initializing simulation at {simulation_folder}")

    if optimize and onGA_hp is None:
        raise ValueError("Optimization was requested, but no hyper parameters were given.")
    elif optimize:
        online_operation_closed_loop(p, previous_simulation_folder)
    else:
        online_operation_open_loop(p, previous_simulation_folder)
    print('Done!')
    return simulation_folder


def online_operation_open_loop(p: OnlineParameters, previous_simulation_folder: Path = None, history_figsize=(16, 5)):
    # Create simulator instance
    simulator = setup_simulator(p, previous_simulation_folder)

    # Disturb to create first network realization
    simulator.disturb_network()
    simulator.save_network()

    # Start loop
    while not simulator.done():
        simulator.forward_fleet()
        simulator.save_history()

    # Save history figures
    simulator.save_history_figures(history_figsize=history_figsize)
    return


def online_operation_closed_loop(p: OnlineParameters, previous_simulation_folder: Path = None, history_figsize=(16, 5)):
    # Create simulator instance
    simulator = setup_simulator(p, previous_simulation_folder)

    # Create dispatcher instance
    dispatcher = Dispatcher.Dispatcher(simulator.network_path, simulator.fleet_path, simulator.measurements_path,
                                       simulator.routes_path, p.ev_type, p.fleet_type, p.edge_type, p.network_type,
                                       p.hyper_parameters)

    exec_time_path = Path(p.simulation_folder, 'exec_time.csv')

    # Disturb to create first network realization
    simulator.disturb_network()
    simulator.save_network()

    # Start loop
    while not simulator.done():
        # Measure and update
        dispatcher.update()
        dispatcher.optimize_online(exec_time_filepath=exec_time_path)

        # Simulate one step
        simulator.forward_fleet()
        simulator.save_history()

    # Save history figures
    simulator.save_history_figures(history_figsize=history_figsize)
    return


def online_operation_degradation(source_folder: Path, main_folder: Path, hp: HyperParameters, eta_table: np.ndarray,
                                 keep_times: int = 4, sample_time: float = 2.0,
                                 std_factor: Tuple[float, float] = (1., 1.), policy=(20., 95.), degrade_until=0.8):
    eta_model = NearestNeighbors(n_neighbors=3).fit(eta_table[:, 0:2])
    now = datetime.datetime.now().strftime("%H-%M-%S_%d-%m-%y")

    policy_folder = Path(main_folder, f'policy_{policy[0]}_{policy[1]}')
    simulation_folder = Path(policy_folder, now)

    day = 1
    fleet_filepath = None
    routes_filepath = None
    previous_day_measurements = None

    while True:
        day_operation_folder = Path(simulation_folder, f'day_{day}')
        simulator = setup_simulator(source_folder, day_operation_folder, sample_time, std_factor, fleet_filepath,
                                    previous_day_measurements=previous_day_measurements,
                                    routes_filepath=routes_filepath,
                                    new_soc_policy=policy)
        simulator.eta_model = eta_model
        simulator.eta_table = eta_table
        dispatcher = Dispatcher.Dispatcher(simulator.network_path, simulator.fleet_path,
                                           simulator.measurements_path, simulator.routes_path,
                                           onGA_hyper_parameters=hp)
        exec_time_path = Path(day_operation_folder, 'exec_time.csv')
        # Start loop
        dont_alter = 0
        while not simulator.done():
            if not dont_alter:
                simulator.disturb_network()
                dispatcher.update()
                dispatcher.optimize_online(exec_time_filepath=exec_time_path)
            dont_alter = dont_alter + 1 if dont_alter + 1 < keep_times else 0
            simulator.forward_fleet()
            simulator.save_history()

        # If degraded, stop
        if not simulator.fleet.healthy(degrade_until):
            break

        # Setup to begin a new day
        fleet_filepath = Path(day_operation_folder, 'fleet_end.xml')
        simulator.fleet.write_xml(fleet_filepath, print_pretty=False)

        routes_filepath = Path(day_operation_folder, 'routes_end.xml')
        simulator.write_routes(routes_filepath)
        previous_day_measurements = simulator.measurements
        day += 1
    return


def setup_simulator(op: OnlineParameters, previous_simulation_folder: Path = None, previous_day_measurements=None, def_cicle='energy'):
    if previous_simulation_folder is not None:
        previous_history_path = Path(previous_simulation_folder, "history.xml")
        previous_measurements_path = Path(previous_simulation_folder, "measurements.xml")

    else:
        previous_history_path = None
        previous_measurements_path = None

    if def_cicle == 'energy':
        sim = Simulator.Simulator(op.simulation_folder, op.network_path, op.fleet_path, op.routes_path, op.sample_time,
                                previous_history_path, previous_measurements_path, op.std_factor, op.eta_model,
                                op.eta_table, op.soc_policy, op.start_earlier_by, previous_day_measurements)
    else:
        sim = SimulatorLV.Simulator(op.simulation_folder, op.network_path, op.fleet_path, op.routes_path, op.sample_time,
                                previous_history_path, previous_measurements_path, op.std_factor, op.eta_model,
                                op.eta_table, op.soc_policy, op.start_earlier_by, previous_day_measurements)
    return sim

"""
MODIFICACION DE LUCKAS: se agrega una funcion para realizar experimento de degradacin-openloop
"""

def online_operation_degradation_open_loop(instance: Path, source_folder: Path, main_folder: Path, hp: HyperParameters, eta_table: np.ndarray,
                                 ev_type: Type[Fleet.EV.ElectricVehicle] = Fleet.EV.ElectricVehicle,
                                 fleet_type: Type[Fleet.Fleet] = Fleet.Fleet,
                                 edge_type: Type[Network.Edge.DynamicEdge] = Network.Edge.DynamicEdge,
                                 network_type: Type[Network.Network] = Network.Network, 
                                 keep_times: int = 4, sample_time: float = 2.0,
                                 std_factor: float = 1.0, policy=(20., 95.), degrade_until=0.8, 
                                 cicle='energy', set_model='aramis', repetitions=5, additional_vehicles=1, fill_up_to=1):

    eta_model = NearestNeighbors(n_neighbors=3).fit(eta_table[:, 0:2])
    now = datetime.datetime.now().strftime("%H-%M-%S_%d-%m-%y")

    policy_folder = Path(main_folder, f'policy_{policy[0]}_{policy[1]}')
    simulation_folder = Path(policy_folder, now+'_'+set_model+'_'+cicle)

    day = 1
    fleet_filepath = None
    routes_filepath = None
    previous_day_measurements = None

    day_operation_folder = Path(simulation_folder, 'day_data')

    op = False
    kt = 4
    seb = 600
    #p = OnlineParameters(source_folder, day_operation_folder, op, policy, hp, kt, sample_time,
    #                     std_factor, seb, ev_type, fleet_type, edge_type, network_type, eta_model=eta_model, eta_table=eta_table)
    
    previous_simulation_folder = None

    while True:
        p = OnlineParameters(source_folder, day_operation_folder, op, policy, hp, kt, sample_time,
                            std_factor, seb, ev_type, fleet_type, edge_type, network_type, eta_model=eta_model, eta_table=eta_table)
        print(f'Dia {day} en politica {policy[0]} {policy[1]}')
        #day_operation_folder = Path(simulation_folder, f'day_{day}')
        simulator = setup_simulator(p, previous_simulation_folder, previous_day_measurements, def_cicle=cicle)
        #simulator = setup_simulator(source_folder, day_operation_folder, sample_time, std_factor, fleet_filepath,
        #                            previous_day_measurements=previous_day_measurements,
        #                            routes_filepath=routes_filepath,
        #                            new_soc_policy=policy, def_cicle=cicle)
        simulator.eta_model = eta_model
        simulator.eta_table = eta_table
        #dispatcher = Dispatcher.Dispatcher(simulator.network_path, simulator.fleet_path,
        #                                   simulator.measurements_path, simulator.routes_path,
        #                                   onGA_hyper_parameters=hp)
        exec_time_path = Path(day_operation_folder, 'exec_time.csv')
        # Start loop
        dont_alter = 0
        while not simulator.done():
            if not dont_alter:
                simulator.disturb_network()
                #dispatcher.update()
                #dispatcher.optimize_online(exec_time_filepath=exec_time_path)
            dont_alter = dont_alter + 1 if dont_alter + 1 < keep_times else 0
            simulator.forward_fleet(degradation_model=set_model)
            simulator.save_history()

        # If degraded, stop
        if not simulator.fleet.healthy(degrade_until):
            break

        # Setup to begin a new day
        fleet_filepath = Path(day_operation_folder, 'fleet_end.xml')
        simulator.fleet.write_xml(fleet_filepath, print_pretty=False)
        # Actualizar estado de bateria en fleet_end.xml
        #update_battery(day_operation_folder)

        routes_filepath = Path(day_operation_folder, 'routes_end.xml')
        simulator.write_routes(routes_filepath)
        previous_day_measurements = simulator.measurements
        day += 1

        # Save data from fleet_end, history and measurements into csv
        save_to_csv(day_operation_folder, simulation_folder)
        save_history(day_operation_folder, simulation_folder)

        # Update the instance to realize the pre-operation for the next day
        instances_folder = update_instance(instance, day_operation_folder, instance=True)

        # Realize the pre-operation for the next day
        s = PreOperation.folder_pre_operation(instances_folder, hp, policy, additional_vehicles, fill_up_to, None, 'deterministic', repetitions, fleet_type,
                                          ev_type, network_type, edge_type, 120,
                                          2, False, True)

        # Save the pre-operation data
        save_preoperation(Path(str(s)), simulation_folder)

        # Update previous simulation folder
        previous_simulation_folder = day_operation_folder
        
        # Change source folder to best solution of pre-operation
        source_folder = Path(str(s))
        routes_filepath = None
    return

    '''
MODIFICACION DE LUCKAS: se agrega una funcion para actualizar las instancias dia a dia
'''  
def update_battery(day_operation_folder= Path):
    if os.path.exists(Path( day_operation_folder, 'degradation_event.csv')):
        degradation_event_df = pd.read_csv(Path( day_operation_folder, 'degradation_event.csv'))
        degradation_event_df.drop(degradation_event_df.filter(regex='Unname'), axis=1, inplace=True)

        battery_capacity = degradation_event_df['Capacidad Actual'].iloc[-1]

        tree = ET.parse(Path( day_operation_folder, 'fleet_end.xml'))
        root = tree.getroot()

        for ev in root.iter('electric_vehicle'):
            ev.set('battery_capacity',str(battery_capacity))
            tree.write(Path( day_operation_folder, 'fleet_end.xml'))
    else:
        print('Aun no degrada')

def update_instance(source_folder: Path, day_operation_folder= Path, instance=False):
    try:
        tree = ET.parse(source_folder)
        root = tree.getroot()
    except ET.ParseError:
        raise ET.ParseError(f"Not well-formed at file:\n {source_folder}")

    if instance:
        _fleet = tree.find('fleet')
    else:
        _fleet = tree.getroot()
    
    new_tree = ET.parse(Path(day_operation_folder,'fleet_end.xml'))
    new_root = new_tree.getroot()
    
    root.remove(_fleet)
    root.append(new_root)

    Path(day_operation_folder.parent, 'new_instance').mkdir(parents=True, exist_ok=True)

    new_path = Path(day_operation_folder.parent, 'new_instance', source_folder.name)
    tree.write(new_path)

    return new_path.parent

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
    return data

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

def get_root_and_tree(filepath: Path):
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
    except ET.ParseError:
        raise ET.ParseError(f"Not well-formed at file:\n {filepath}")
    
    return root, tree

def create_etree(root_name: str):
    root = ET.Element(root_name)
    tree = ET.ElementTree(root)
    return root, tree

def save_xml(file_path: Path, root):
    if not os.path.isfile(file_path):
        new_root, new_tree = create_etree('top')
        new_root.append(root)
        new_tree.write(file_path)
    else:
        new_root, new_tree = get_root_and_tree(file_path)
        new_root.append(root)
        new_tree.write(file_path)


def save_preoperation(source_folder: Path, save_folder: Path):
    for file in source_folder.iterdir():
        if file.stem == 'fleet' and file.suffix == '.xml':
            fleet_filepath = file
        elif file.stem == 'network' and file.suffix == '.xml':
            network_filepath = file
        elif file.stem == 'routes' and file.suffix == '.xml':
            routes_filepath = file
        elif file.stem == 'optimization_report' and file.suffix == '.csv':
            report_filepath = file
    
    root_fleet, tree_fleet = get_root_and_tree(fleet_filepath)
    root_network, tree_network = get_root_and_tree(network_filepath)
    root_routes, tree_routes = get_root_and_tree(routes_filepath)

    fleet_path = Path(save_folder, 'preoperations_fleet.xml')
    network_path = Path(save_folder, 'preoperations_network.xml')
    routes_path = Path(save_folder, 'preoperations_routes.xml')
    report_path = Path(save_folder, 'preoperations_report.csv')

    save_xml(fleet_path, root_fleet)
    #save_xml(network_path, root_network)
    save_xml(routes_path, root_routes)

    report = pd.read_csv(report_filepath)
    report.rename({'Unnamed: 0':'Col', '0':'Val'}, axis=1, inplace=True)
    keys, val = list(report['Col']), list(report['Val'])
    res = {keys[i]: val[i] for i in range(len(keys))}
    day_report = pd.DataFrame(res, index= [0])

    if not os.path.isfile(report_path):
        day_report.to_csv(report_path)
    else:
        previous_report = pd.read_csv(report_path)
        previous_report.drop(previous_report.filter(regex='Unname'), axis=1, inplace=True)
        previous_report.reset_index(inplace=True, drop=True)
        full_report = pd.concat([previous_report, day_report], axis=0, ignore_index=True)
        full_report.to_csv(report_path)

def save_history(source_folder: Path, save_folder: Path):
    for file in source_folder.iterdir():
        if file.stem == 'history' and file.suffix == '.xml':
            fleet_filepath = file
    
    root_history, tree_history = get_root_and_tree(fleet_filepath)

    history_path = Path(save_folder, 'history.xml')

    save_xml(history_path, root_history)

