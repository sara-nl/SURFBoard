import argparse
import pandas as pd
import glob
from collections import namedtuple
from xml.dom import minidom
from collections import defaultdict, namedtuple

class ProfileParser():

    def __init__(self, file_name):
        self.gpus_data = {}
        self.threads_data = {}
        self._create_xml_dom(file_name)
        self._parse_thread_data()

    def _create_xml_dom(self, file_name):
        '''
        Open the xml profile and store the xml tree in self.xml.
        Called on init.
        '''
        malformed_tau_xml = open(file_name, encoding="utf-8")
        malformed_xml_str = malformed_tau_xml.read().replace(u"\x00", "")
        malformed_tau_xml.close()
        # we add a fake root, so that the parser works
        good_xml_str = f"<fakeroot>\n{malformed_xml_str}</fakeroot>\n"
        self.xml = minidom.parseString(good_xml_str)

    def _parse_thread_data(self):

        # init threads_data
        threads = self.get_threads()
        for thread in threads:
            if thread.id not in self.threads_data:
                self.threads_data[thread.id] = {'interval': [], 'atomic': []}

        # init mapping between thread id and profile DOM element for fast lookup
        profiles_map = {}
        profiles = self.xml.getElementsByTagName("profile")
        for profile in profiles:
            thread = profile.getAttribute("thread")
            if thread not in profiles_map:
                profiles_map[thread] = profile

        # get list of gpus ids (from 0 to gpu-count - 1)
        threads = self.get_threads()
        for thread in threads:
            profile = profiles_map[thread.id]

            # interval data
            interval_data = profile.getElementsByTagName("interval_data")[0].firstChild.nodeValue.strip()
            interval_data_lines = self._parse_interval_data(interval_data)

            events = self.xml.getElementsByTagName("event")
            id_event = defaultdict(str)

            for event in events:
                id = int(event.getAttribute('id'))

                name = event.getElementsByTagName("name")[0].firstChild.nodeValue
                id_event[id] = self._format_mpi_name(name)

            Function = namedtuple('Function', ['id', 'name', 'calls', 'subcalls', 'exc_time', 'inc_time'])
            for line in interval_data_lines:
                id = line[0]
                if id in id_event:
                    f = Function(id, id_event[id], line[1], line[2], line[3], line[4])
                    self.threads_data[thread.id]['interval'].append(f)

            # atomic data
            atomic_data = profile.getElementsByTagName("atomic_data")[0].firstChild.nodeValue.strip()
            atomic_data_lines = self._parse_atomic_data(atomic_data)

            user_events = self.xml.getElementsByTagName("userevent")
            user_event_id = defaultdict(str)

            for user_event in user_events:
                id = int(user_event.getAttribute('id'))
                name = user_event.getElementsByTagName('name')[0].firstChild.nodeValue
                user_event_id[id] = name

            Metric = namedtuple('Metric', ['id', 'name', 'numSamples', 'max', 'min', 'mean', 'unknown'])
            for line in atomic_data_lines:
                id = line[0]
                if id in user_event_id:
                    m = Metric(id, user_event_id[id], line[1], line[2], line[3], line[4], line[5])
                    self.threads_data[thread.id]['atomic'].append(m)

    def _parse_interval_data(self, interval_data):
        return list(map(lambda x: list(map(int, x.split())), interval_data.split('\n')))

    def _parse_atomic_data(self, atomic_data):
        def _convert(line):
            l = line.split()
            return [int(l[0]), int(l[1]), int(l[2]), int(l[3]), float(l[4]), float(l[5])]
        return list(map(_convert, atomic_data.split('\n')))

    def _format_mpi_name(self, function_name):
        func_name_lower = function_name.lower()
        if func_name_lower[-2:] == "()":
            return func_name_lower[:-2] # remove parenthesis
        else:
            return func_name_lower

    def get_threads(self):
        '''
        Return a list of namedtuple('Thread', ["id", "gpu", "context", "thread"])
        Each namedtuple correspond to a unique Thread function and info such as
        id (IP address),
        gpu (GPU number, threads with same GPU number are running on the same GPU),
        context ([not sure what this correspond to]),
        thread (if 1, then the thread is an 'MPI thread', 0 otherwise)
        '''
        threads = self.xml.getElementsByTagName("thread")
        Thread = namedtuple("Thread", ["id", "gpu", "context", "thread"])
        threads_info = set()
        for thread in threads:
            id, gpu, context, thread = thread.attributes.items()
            threads_info.add(Thread(id[1], int(gpu[1]), int(context[1]), int(thread[1])))
        return sorted(threads_info, key=lambda thread: (thread.gpu, thread.thread))
    
    def __repr__(self):
        r = ''
        for gpu, data in self.gpus_data.items():
            r += f'GPU #{gpu}\n'
            for type, l in data.items():
                r += f'\t{type}\n'
                for e in l:
                    r += f'\t\t{e}\n'
        return r


def get_data(filename):
    parser = ProfileParser(filename)

    Interval = namedtuple('Interval', ['gpu', 'thread', 'function', 'calls', 'subcalls', 'exc_time', 'inc_time'])
    interval_functions = []
    Atomic = namedtuple('Atomic', ['gpu', 'thread', 'function', 'num_samples', 'max', 'min', 'mean'])
    atomic_functions = []

    for thread, data in parser.threads_data.items():
        gpu = thread.split('.')[0]
        thread = thread.split('.')[2]

        for f in data['interval']:
            interval_list = [gpu, thread] + list(f)[1:]
            interval_functions.append(Interval(*interval_list))

        for f in data['atomic']:
            atomic_list = [gpu, thread] + list(f)[1:-1]
            atomic_functions.append(Atomic(*atomic_list))

    return interval_functions, atomic_functions

def convert_to_csv(file, name):
    print("Convertion of TAU xml profile to csv started.")
    interval_data, atomic_data = get_data(file)

    df_interval = pd.DataFrame(
        interval_data,
        columns=[
            'gpu',
            'thread',
            'function',
            'calls',
            'subcalls',
            'exc_time',
            'inc_time',
            ]
        )
    df_interval.to_csv(f'{name}_interval.csv', index=False)

    df_atomic = pd.DataFrame(
        atomic_data,
        columns=[
            'gpu',
            'thread',
            'function',
            'num_samples',
            'max',
            'min',
            'mean',
            ]
        )
    df_atomic.to_csv(f'{name}_atomic.csv', index=False)
    print("Convertion of TAU xml profile to csv ended.")

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", help="Path to the xml profile.")
    parser.add_argument("--output", "-o", help="Ouputs two files: {output}_interval.csv and {output}_atomic.csv.", default="tauprofile")
    return parser.parse_args()

def main():
    parser = parse_arguments()
    convert_to_csv(parser.xml, parser.output)

if __name__ == "__main__":
    main()
