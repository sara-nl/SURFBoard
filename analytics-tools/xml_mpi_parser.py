import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import glob
from xml.dom import minidom
import xmltodict
from collections import defaultdict, namedtuple
import argparse
import sys
import operator


class ProfileParser():

    _mpi_id_func = None
    mpi_func_names = None
    mpi_data = None

    def __init__(self, file_name):
        self.file_name = file_name
        self._create_xml_dom(file_name)

    def _create_xml_dom(self, file_name):
        '''
        Open the xml profile and store the xml tree in self.xml.
        Called on init.
        '''
        malformed_tau_xml = open(self.file_name, encoding="utf-8")
        malformed_xml_str = malformed_tau_xml.read().replace(u"\x00", "")
        malformed_tau_xml.close()
        # we add a fake root, so that the parser works
        good_xml_str = f"<fakeroot>\n{malformed_xml_str}</fakeroot>\n"
        self.xml = minidom.parseString(good_xml_str)

    def get_mpi_func_names(self):
        '''
        Return a list of all the MPI functions present in the xml profile.
        '''
        # if mpi_func_names is not None, it was already initialized
        if self.mpi_func_names is not None:
            return self.mpi_func_names

        # if mpi_id_func is not None, get the list of function names from it
        if self._mpi_id_func is not None:
            self.mpi_func_names = [func for id, func in self._mpi_id_func]
            return self.mpi_func_names

        # otherwise, initialize it
        events = self.xml.getElementsByTagName("event")
        mpi_id_func = defaultdict(str)
        for event in events:
            id = int(event.getAttribute('id'))
            group = event.getElementsByTagName("group")[0].firstChild.nodeValue
            if group == 'MPI':
                name = event.getElementsByTagName("name")[0].firstChild.nodeValue
                mpi_id_func[id] = self._format_mpi_name(name)
        self._mpi_id_func = dict(mpi_id_func)
        self.mpi_func_names = list(mpi_id_func.values())
        return self.mpi_func_names

    def get_mpi_data(self):
        '''
        Return a list of namedtuple('Function', ['id', 'name', 'calls', 'subcalls', 'exc_time', 'inc_time']
        Each namedtuple correspond to a unique MPI function and contains metrics
        such as number of event id, name calls, number of subcalls, excluded
        execution time, included execution time.
        '''
        if self.mpi_data is not None:
            return self.mpi_data

        derivedinterval_data = ""
        for derivedprofile in self.xml.getElementsByTagName("derivedprofile"):
            if derivedprofile.getAttribute("derivedentity") == "total":
                derivedinterval_data = derivedprofile.getElementsByTagName("derivedinterval_data")[0].firstChild.nodeValue.strip()
                break

        # convert data from string to list of lines where each line is a list of int
        derivedinterval_data_lines = list(map(lambda x: list(map(int, x.split())), derivedinterval_data.split('\n')))

        mpi_functions = []
        Function = namedtuple('Function', ['id', 'name', 'calls', 'subcalls', 'exc_time', 'inc_time'])
        for line in derivedinterval_data_lines:
            if line[0] in self._mpi_id_func.keys():
                function = Function(line[0], self._mpi_id_func[line[0]], line[1], line[2], line[3], line[4])
                mpi_functions.append(function)
        self.mpi_data = mpi_functions
        return mpi_functions

    def get_mpi_metrics(self, mpi_function_name):
        for mpi_function in self.get_mpi_data():
            if mpi_function.name == mpi_function_name:
                return mpi_function
        print(f"Did not find mpi_function: '{self._format_mpi_name(function_name)}'")
        return -1

    def get_mpi_calls(self, mpi_function_name):
        for mpi_function in self.get_mpi_data():
            if mpi_function.name == mpi_function_name:
                return mpi_function.calls
        print(f"Did not find mpi_function: '{self._format_mpi_name(function_name)}'")
        return -1

    def get_mpi_subcalls(self, mpi_function_name):
        for mpi_function in self.get_mpi_data():
            if mpi_function.name == mpi_function_name:
                return mpi_function.subcalls
        print(f"Did not find mpi_function: '{self._format_mpi_name(function_name)}'")
        return -1

    def get_mpi_exc_time(self, mpi_function_name):
        for mpi_function in self.get_mpi_data():
            if mpi_function.name == mpi_function_name:
                return mpi_function.exc_time
        print(f"Did not find mpi_function: '{self._format_mpi_name(function_name)}'")
        return -1

    def get_mpi_inc_time(self, mpi_function_name):
        for mpi_function in self.get_mpi_data():
            if mpi_function.name == mpi_function_name:
                return mpi_function.inc_time
        print(f"Did not find mpi_function: '{self._format_mpi_name(function_name)}'")
        return -1

    def get_mpi_event_id(self, mpi_function_name):
        for mpi_function in self.get_mpi_data():
            if mpi_function.name == mpi_function_name:
                return mpi_function.id
        print(f"Did not find mpi_function: '{self._format_mpi_name(function_name)}'")
        return -1

    def _format_mpi_name(self, function_name):
        func_name_lower = function_name.lower()
        if func_name_lower[-2:] == "()":
            return func_name_lower[:-2] # remove parenthesis
        else:
            return func_name_lower


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="Path to the directory containing the xml profiles.")
    return parser.parse_args()


def get_files(data_dir):
    return glob.glob(f"{data_dir}*.xml")


def plot_top_mpi_exc_times(exc_times_mpi):

    median_exc_times_mpi = [(func, np.median(exc_times)) for func, exc_times in exc_times_mpi.items()]
    sorted_median_exc_times_mpi = sorted(median_exc_times_mpi, key=operator.itemgetter(1))
    sorted_mpi_functions = [func for func, median_exc_times in sorted_median_exc_times_mpi]
    sorted_mpi_median_exc_times = [median_exc_times for func, median_exc_times in sorted_median_exc_times_mpi]

    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (14, 7))
    plt.barh(sorted_mpi_functions, sorted_mpi_median_exc_times)
    plt.yticks(sorted_mpi_functions)
    plt.xticks(range(0, 460, 50))
    plt.xlabel('time [s]')
    plt.title('median excluded time for each MPI function')
    plt.show()


def main():
    parser = parse_arguments()
    files = get_files(parser.data_dir)

    exc_times_mpi = defaultdict(list)
    for file in files:
        file_parser = ProfileParser(file)
        for func in file_parser.get_mpi_func_names():
            exc_times_mpi[func].append(file_parser.get_mpi_exc_time(func))

    plot_top_mpi_exc_times(exc_times_mpi)


main()
