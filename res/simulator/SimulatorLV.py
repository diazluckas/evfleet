from os import makedirs
from pathlib import Path
from typing import Tuple, Union, List

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

import res.dispatcher.Dispatcher as Dispatcher
import res.models.Fleet as Fleet
import res.models.Network as Network
import res.simulator.History as History


def disturb_network(network_source: Network.Network, network: Network.Network, std_factor=1.):
    for i in network_source.nodes.keys():
        for j in network_source.nodes.keys():
            edge_source = network_source.edges[i][j]
            edge_target = network.edges[i][j]
            realization = np.random.normal(loc=edge_source.velocity, scale=std_factor * edge_source.velocity_deviation)
            edge_target.velocity = realization


def saturate(val, min_val=-np.infty, max_val=np.infty):
    if val < min_val:
        val = min_val
    if val > max_val:
        val = max_val
    return val


class Simulator:
    def __init__(self, operation_folder: Path, network_path: Path, fleet_path: Path, routes_path: Path,
                 sample_time: float = 300., history_path: Path = None, measurements_path: Path = None,
                 std_factor: float = 1.0, eta_model: NearestNeighbors = None, eta_table: np.ndarray = None,
                 new_soc_policy: Tuple[Union[float, int], Union[float, int]] = None,
                 start_earlier_by: Union[float, int] = 0):

        # Setup directories
        self.main_folder = operation_folder

        self.network_path_original = network_path
        self.network_path = Path(operation_folder, 'network_temp.xml')

        self.fleet_path_original = fleet_path
        self.fleet_path = Path(operation_folder, 'fleet_temp.xml')

        self.routes_path_original = routes_path
        self.routes_path = Path(operation_folder, 'routes_temp.xml')

        self.measurements_path_original = measurements_path
        self.measurements_path = Path(operation_folder, "measurements.xml")

        self.history_path_original = history_path
        self.history_path = Path(operation_folder, "history.xml")

        self.history_figure_path = Path(operation_folder)

        # Setup sample time
        self.sample_time = sample_time

        # Create main simulation directory
        makedirs(operation_folder, exist_ok=True)

        # Read network, fleet and route files
        self.network_source = Network.GaussianCapacitatedNetwork.from_xml(self.network_path_original, False,
                                                                          Network.Edge.GaussianEdge)
        self.fleet_source = Fleet.GaussianFleet.from_xml(self.fleet_path_original, False)

        self.network = Network.GaussianCapacitatedNetwork.from_xml(self.network_path_original, False,
                                                                   Network.Edge.GaussianEdge)
        self.fleet = Fleet.GaussianFleet.from_xml(self.fleet_path_original, False)
        self.routes, self.depart_info = Dispatcher.read_routes(routes_path, read_depart_info=True)

        if new_soc_policy:
            self.fleet.new_soc_policy(new_soc_policy[0], new_soc_policy[1])
            self.depart_info = {id_ev: (i[0], new_soc_policy[1], i[2]) for id_ev, i in self.depart_info.items()}

        # Save temporary network, fleet and route files
        self.network.write_xml(self.network_path)
        self.fleet.write_xml(self.fleet_path)
        Dispatcher.write_routes(self.routes_path, self.routes, self.depart_info)

        # Create or read measurements file
        self.measurements, _ = self.create_measurements_file(start_earlier_by)

        # Create or read history file
        self.history = History.FleetHistory().create_from_routes(self.routes)
        self.save_history()

        # Setup other variables
        self.std_factor = std_factor
        self.day_points = int(24 * 60 * 60 / self.network.edges[0][0].sample_time)
        self.eta_model = eta_model
        self.eta_table = eta_table

    def create_measurements_file(self, start_earlier_by: float = 0.0):
        return Dispatcher.create_measurements_file(self.measurements_path, self.routes_path, start_earlier_by,
                                                   self.measurements_path_original)

    def save_history(self):
        self.history.save(self.history_path, write_pretty=False)

    def disturb_network(self):
        disturb_network(self.network_source, self.network, std_factor=self.std_factor)

    def disturb_edge(self, node_from: int, node_to: int):
        self.network.edges[node_from][node_to].disturb()

    def disturb_edges(self, node_sequence: Union[List[int], Tuple[int, ...]], probability=0.1):
        if len(node_sequence) <= 1:
            return
        for i, j in zip(node_sequence[:-1], node_sequence[1:]):
            if np.random.uniform() <= probability:
                self.network.edges[i][j].disturb()

    def save_network(self):
        self.network.write_xml(self.network_path)

    def write_measurements(self):
        Dispatcher.write_measurements(self.measurements_path, self.measurements, write_pretty=False)

    def write_routes(self, path: Path = None):
        depart_info = {}
        routes = {}
        for id_ev, m in self.measurements.items():
            m = self.measurements[id_ev]
            depart_info[id_ev] = (m.departure_time, self.depart_info[id_ev][1], self.depart_info[id_ev][2])
            route_no_departure_wt = (0.,) + self.routes[id_ev][2][1:]
            routes[id_ev] = (self.routes[id_ev][0], self.routes[id_ev][1], route_no_departure_wt)

        if path:
            Dispatcher.write_routes(path, routes, depart_info=depart_info, write_pretty=False)
        else:
            Dispatcher.write_routes(self.routes_path, routes, depart_info=depart_info, write_pretty=False)

    def update_routes(self, read_depart_info=False):
        if read_depart_info:
            self.routes, self.depart_info = Dispatcher.read_routes(self.routes_path, read_depart_info)
        else:
            self.routes, _ = Dispatcher.read_routes(self.routes_path, read_depart_info)

    def forward_node_service(self, measurement: Dispatcher.ElectricVehicleMeasurement, simulation_step: float,
                             S: Tuple[int, ...], L: [float, ...], wt1: Tuple[float, ...], k: int,
                             ev: Fleet.EV.ElectricVehicle, additive_noise_gain=0.0):
        """
        Simulates the at-node service of one EV
        @param measurement: EV measurement instance containing current measurements
        @param simulation_step: the simulation step in seconds
        @param S: destinations sequence of the EV
        @param L: recharging plan of the EV
        @param wt1: post-service waiting times of the EV
        @param k: index of the current node the EV is at in S
        @param ev: EV instance
        @return: EV measurement with simulation done
        """
        Sk = S[k]
        current_time = measurement.time
        eos_time = measurement.time_finishing_service
        post_waiting_time = wt1[k]

        future_time = current_time + simulation_step
        departure_time = eos_time + post_waiting_time

        # Vehicle departs from the node
        if future_time >= departure_time:
            departure_soc = measurement.soc_finishing_service
            departure_payload = measurement.payload_finishing_service

            measurement.time = departure_time
            measurement.soc = departure_soc
            measurement.payload = departure_payload

            measurement.stopped_at_node_from = False

            # measurement.max_soc = departure_soc if departure_soc > measurement.max_soc else measurement.max_soc
            measurement.update_max_soc(departure_soc)

            # EV departs from the depot: record this time
            if not Sk:
                measurement.departure_time = departure_time

            # Check constraints violations
            #if departure_time > self.network.time_window_upp(Sk):
            #    v = History.ConstraintViolation('time_window_upp', self.network.time_window_upp(Sk), departure_time, Sk,
            #                                    'leaving')
            #    self.history.add_vehicle_constraint_violation(ev.id, v)

            if ev.alpha_up < measurement.soc_finishing_service:
                v = History.ConstraintViolation('alpha_up', ev.alpha_up, departure_soc, Sk, 'finishing_service')
                self.history.add_vehicle_constraint_violation(ev.id, v)

            if ev.alpha_down > measurement.soc_finishing_service:
                v = History.ConstraintViolation('alpha_down', ev.alpha_down, departure_soc, Sk, 'finishing_service')
                self.history.add_vehicle_constraint_violation(ev.id, v)

            # Add event
            event = History.NodeEvent(False, eos_time, departure_soc, departure_payload, Sk,
                                      post_service_waiting_time=post_waiting_time)
            self.history.add_vehicle_event(ev.id, event)

            # Disturb remaining route
            self.disturb_edges(S[k:], probability=0.3)

            # Continue recursive iteration
            next_step_size = future_time - departure_time
            if next_step_size > 0:
                self.forward_vehicle(ev.id, next_step_size, additive_noise_gain=additive_noise_gain, model='aramis')

        # Vehicle stays at the node
        else:
            measurement.time += simulation_step
        pass

    def forward_arc_travel(self, measurement: Dispatcher.ElectricVehicleMeasurement, simulation_step: float,
                           S: Tuple[int, ...], L: [float, ...], wt1: Tuple[float, ...], k: int,
                           ev: Fleet.EV.ElectricVehicle, additive_noise_gain: float = 0.0, deg_model='aramis'):
        S0 = S[k]
        S1 = S[k + 1]
        current_time = measurement.time
        eos_time_at_S0 = measurement.time_finishing_service
        post_waiting_time_at_S0 = wt1[k]

        future_time = current_time + simulation_step
        departure_time_from_S0 = eos_time_at_S0 + post_waiting_time_at_S0

        tij = self.network.t(S0, S1, departure_time_from_S0, additive_noise_gain=additive_noise_gain)
        Eij = self.network.E(S0, S1, measurement.payload + ev.weight, departure_time_from_S0,
                             additive_noise_gain=additive_noise_gain)
        Kmij = self.network.km(S0, S1, departure_time_from_S0, additive_noise_gain=additive_noise_gain)

        eij = 100 * Eij / ev.battery_capacity
        delta = simulation_step / tij if tij else 1.0
        delta = delta if delta < 1 - measurement.eta else 1 - measurement.eta

        tij_traveled = tij * delta
        Eij_traveled = Eij * delta
        eij_traveled = eij * delta

        measurement.cumulated_consumed_energy += Eij_traveled

        # Vehicle reaches next node
        if delta >= 1 - measurement.eta:
            arrival_time = current_time + tij_traveled
            arrival_soc = measurement.soc - eij_traveled
            arrival_payload = measurement.payload

            measurement.visited_nodes += 1
            measurement.clients_visited += 1
            measurement.km += int(Kmij)
            measurement.stopped_at_node_from = True

            # Check constraints violations when arriving
            if arrival_soc > ev.alpha_up:
                v = History.ConstraintViolation('alpha_up', ev.alpha_up, arrival_soc, S1, 'arriving')
                self.history.add_vehicle_constraint_violation(ev.id, v)

            if arrival_soc < ev.alpha_down:
                v = History.ConstraintViolation('alpha_down', ev.alpha_down, arrival_soc, S1, 'arriving')
                self.history.add_vehicle_constraint_violation(ev.id, v)

            # The node is not the depot
            if S1:
                S2 = S[k + 2]
                tw_low = self.network.nodes[S1].time_window_low
                waiting_time = tw_low - arrival_time if tw_low > arrival_time else 0.

                soc_increase = L[k + 1] if arrival_soc + L[k + 1] <= ev.alpha_up else ev.alpha_up - arrival_soc
                service_time = self.network.spent_time(S1, arrival_soc, soc_increase)

                eos_time = arrival_time + waiting_time + service_time
                eos_soc = arrival_soc + soc_increase
                eos_payload = arrival_payload - self.network.demand(S1)

                if self.network.is_charging_station(S1):
                    measurement.clients_visited -= 1
                    price = self.network.nodes[S1].price
                    self.history.update_recharging_cost(ev.id, price * (soc_increase/100) * (ev.battery_capacity/3600000))
                    self.history.update_recharging_time(ev.id, service_time)
                    measurement.cumulated_charged_energy += (soc_increase/100) * (ev.battery_capacity/3600000)

                    # Degrade if an energy consumption equal to the nominal capacity is reached
                    #eta_val = ev.degrade_battery(self.eta_table, self.eta_model, measurement.min_soc, measurement.max_soc)
                    deg_val = ev.degrade_battery(self.eta_table, self.eta_model, measurement.min_soc, measurement.max_soc, 
                                                consumed_energy=measurement.cumulated_consumed_energy, 
                                                charge_energy= measurement.cumulated_charged_energy, set_model=deg_model)
                    # Save this data
                    if deg_model == 'aramis':
                        data = {'time': [measurement.time], 'eta': [deg_val], 'min soc': [measurement.min_soc],
                                'max soc': [measurement.max_soc], 'Energy Consumed': [measurement.cumulated_consumed_energy], 
                        'Energy Charged': [measurement.cumulated_charged_energy]}
                    else:
                        data = {'time': [measurement.time], 'severity_factor': [deg_val], 'min soc': [measurement.min_soc],
                                'max soc': [measurement.max_soc], 'Energy Consumed': [measurement.cumulated_consumed_energy], 
                        'Energy Charged': [measurement.cumulated_charged_energy]}
                    path = Path(self.main_folder, 'degradation_event.csv')
                    with_columns = False if path.is_file() else True
                    with open(path, 'a') as file:
                        pd.DataFrame(data).to_csv(file, index=False, header=with_columns)

                    # Reset value
                    measurement.cumulated_consumed_energy = 0
                    measurement.cumulated_charged_energy = 0
                    measurement.min_soc = measurement.soc
                    measurement.max_soc = measurement.soc

            # The node is the depot: the tour ends
            else:
                S2 = 0
                waiting_time = 0.0
                eos_time = arrival_time
                eos_soc = arrival_soc
                eos_payload = arrival_payload

                measurement.done = True
                measurement.clients_visited -= 1

                measurement.cumulated_charged_energy += ((ev.alpha_up-arrival_soc)/100) * (ev.battery_cpacity/3600000)
                self.history.update_recharging_cost(ev.id, 250 * ((ev.alpha_up-arrival_soc)/100) * (ev.battery_capacity/3600000))

                if arrival_time - measurement.departure_time > ev.max_tour_duration:
                    v = History.ConstraintViolation('max_tour_time', ev.max_tour_duration,
                                                    arrival_time - measurement.departure_time, 0, 'arriving')
                    self.history.add_vehicle_constraint_violation(ev.id, v)
                
                # Degrade if an energy consumption equal to the nominal capacity is reached
                #eta_val = ev.degrade_battery(self.eta_table, self.eta_model, measurement.min_soc, measurement.max_soc)
                deg_val = ev.degrade_battery(self.eta_table, self.eta_model, measurement.min_soc, measurement.max_soc, 
                                            consumed_energy=measurement.cumulated_consumed_energy, 
                                            charge_energy= measurement.cumulated_charged_energy, set_model=deg_model)
                # Save this data
                if deg_model == 'aramis':
                    data = {'time': [measurement.time], 'eta': [deg_val], 'min soc': [measurement.min_soc],
                            'max soc': [measurement.max_soc], 'Energy Consumed': [measurement.cumulated_consumed_energy], 
                    'Energy Charged': [measurement.cumulated_charged_energy]}
                else:
                    data = {'time': [measurement.time], 'severity_factor': [deg_val], 'min soc': [measurement.min_soc],
                            'max soc': [measurement.max_soc], 'Energy Consumed': [measurement.cumulated_consumed_energy], 
                    'Energy Charged': [measurement.cumulated_charged_energy]}
                path = Path(self.main_folder, 'degradation_event.csv')
                with_columns = False if path.is_file() else True
                with open(path, 'a') as file:
                    pd.DataFrame(data).to_csv(file, index=False, header=with_columns)

                # Reset value
                measurement.cumulated_consumed_energy = 0
                measurement.cumulated_charged_energy = 0
                measurement.min_soc = measurement.soc
                measurement.max_soc = measurement.soc

            measurement.node_from = S1
            measurement.node_to = S2
            measurement.time_finishing_service = eos_time
            measurement.soc_finishing_service = eos_soc
            measurement.payload_finishing_service = eos_payload
            measurement.time = arrival_time
            measurement.soc = arrival_soc
            measurement.payload = arrival_payload
            measurement.update_min_soc(arrival_soc)
            measurement.eta = 0

            event = History.NodeEvent(True, arrival_time, arrival_soc, arrival_payload, S1, waiting_time)
            self.history.add_vehicle_event(ev.id, event)

            next_step_size = future_time - arrival_time

        # Vehicle continues traversing the arc
        else:
            measurement.time += tij_traveled
            measurement.soc -= eij_traveled
            measurement.eta += delta

            next_step_size = 0.

        """
        # Degrade if an energy consumption equal to the nominal capacity is reached
        if self.eta_model and measurement.cumulated_consumed_energy > ev.battery_capacity_nominal:
            #eta_val = ev.degrade_battery(self.eta_table, self.eta_model, measurement.min_soc, measurement.max_soc)
            deg_val = ev.degrade_battery(self.eta_table, self.eta_model, measurement.min_soc, measurement.max_soc, 
                                        consumed_energy=measurement.cumulated_consumed_energy, 
                                        charge_energy= measurement.cumulated_charged_energy, set_model=deg_model)
            # Save this data
            if deg_model == 'aramis':
                data = {'time': [measurement.time], 'eta': [deg_val], 'min soc': [measurement.min_soc],
                        'max soc': [measurement.max_soc]}
            else:
                data = {'time': [measurement.time], 'severity_factor': [deg_val], 'min soc': [measurement.min_soc],
                        'max soc': [measurement.max_soc]}
            path = Path(self.main_folder, 'degradation_event.csv')
            with_columns = False if path.is_file() else True
            with open(path, 'a') as file:
                pd.DataFrame(data).to_csv(file, index=False, header=with_columns)

            # Reset value
            measurement.cumulated_consumed_energy -= ev.battery_capacity
            measurement.cumulated_charged_energy = 0
            measurement.min_soc = measurement.soc
            measurement.max_soc = measurement.soc
        """
        self.history.update_travelled_time(ev.id, tij_traveled)
        self.history.update_consumed_energy(ev.id, Eij_traveled)

        if next_step_size > 0:
            self.forward_vehicle(ev.id, next_step_size, additive_noise_gain=additive_noise_gain, model='aramis')
        return

    def forward_vehicle(self, id_ev: int, step_time: float = None, additive_noise_gain: float = 0.0, model='aramis'):
        # Select measurement
        measurement = self.measurements[id_ev]

        # If vehicle finished the operation, do nothing
        if measurement.done:
            return

        # If forward time not passed, set it and iterate from there
        if step_time is None:
            step_time = self.sample_time

        # Relevant variables
        ev = self.fleet.vehicles[id_ev]
        route = self.routes[id_ev]
        S = route[0]
        L = route[1]
        wt1 = route[2]
        k0 = measurement.visited_nodes - 1

        # Vehicle is stopped at a node
        if measurement.stopped_at_node_from:
            self.forward_node_service(measurement, step_time, S, L, wt1, k0, ev,
                                      additive_noise_gain=additive_noise_gain)

        # Vehicle is traversing an an arc
        else:
            self.forward_arc_travel(measurement, step_time, S, L, wt1, k0, ev, additive_noise_gain=additive_noise_gain, deg_model=model)

    def forward_fleet(self, additive_noise_gain=0.0, degradation_model='aramis'):
        self.update_routes()
        for id_ev in self.measurements.keys():
            self.forward_vehicle(id_ev, additive_noise_gain=additive_noise_gain, model=degradation_model)
        self.write_measurements()
        self.save_network()

    def save_history_figures(self, history_figsize=(16, 5)):
        figs, info = self.history.draw_events(self.network_source, self.fleet_source, figsize=history_figsize)
        [f.savefig(Path(self.history_figure_path, f"real_operation_EV{i[0]}")) for f, i in zip(figs, info)]
        [f.savefig(Path(self.history_figure_path, f"real_operation_EV{i[0]}.pdf")) for f, i in zip(figs, info)]
        History.plt.close('all')

    def done(self):
        for m in self.measurements.values():
            if not m.done:
                return False
        return True
