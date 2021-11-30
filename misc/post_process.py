import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.special
import scipy.optimize
import scipy.interpolate
import colorsys
import networkx as nx
import sys


class PostProcess:

    def __init__(self, rootname, loads, simulation_type):
        self.root = rootname
        self.loads = loads
        self.N = sum(loads)
        self.n_runs = 18
        self.funcs = [2, 6]
        self.max_reaction_number = min(np.array(loads) * np.array(self.funcs))

        self.monomer_tags = []
        self.reactions = []
        self.cyclenumber = 0
        self.AM = []

        self.radicals0 = 0
        self.pdbs0 = 0

        self.simulation_type = simulation_type
        self.out_file_name = "PolymerNetworkBuilder.pl.out"
        self.set_summary_path = "d:/CAM/PhD_Year2/Networks/3Dprinting/NVP/NVP_keys.txt"
        self.run_folder_rootname = "run"
        self.chdir = "d:/CAM/Phd_Year2/Networks/3Dprinting/NVP"
        self.reaction_names = ["Primary", "Secondary"]

    def file_len(self, fname):
        with open(fname) as f:
            for i, l in enumerate(f):
                pass
        return i + 1

    def load_monomer_tags(self, code_file_path):

        f = open(code_file_path, 'r')
        l = self.file_len(code_file_path)

        c = 0
        atoms = []
        n_atoms = 0
        while (c < l):
            atoms.append([])
            line = f.readline().split()
            n_atoms += len(line)
            for atom_id in line:
                atoms[-1].append(int(atom_id))
            c += 1
        atoms = np.ravel(atoms)
        monomer_tags = np.zeros(n_atoms)

        for i in range(len(atoms)):
            for j in range(len(atoms[i])):
                monomer_tags[atoms[i][j]] += i

        self.monomer_tags = monomer_tags
        return monomer_tags

    def load_reactions(self, outfile_path):
        reactions = []
        f = open(outfile_path, 'r')
        l = self.file_len(outfile_path)

        if (self.simulation_type == "MD"):
            if (len(self.monomer_tags) == 0):
                monomer_tags = self.load_monomer_tags(self.set_summary_path)
            else:
                monomer_tags = self.monomer_tags

            c = 0
            reaction_label = self.reaction_names
            while (c < l):
                line = f.readline().split()
                c += 1
                search = False
                for word in line:
                    for label in reaction_label:
                        if (word == label):
                            search = True
                if (search):
                    run = True
                    while (run):
                        if (len(line) == 0):
                            run = False
                        else:
                            for label in reaction_label:
                                if (line[0] == label and len(line) == 5):
                                    reactions.append([])
                                    atom1 = int(line[2])
                                    atom2 = int(line[4])
                                    reactions[-1].append(int(monomer_tags[atom1]))
                                    reactions[-1].append(int(monomer_tags[atom2]))
                        line = f.readline().split()
                        c += 1

        if (self.simulation_type == "graph"):
            f.readline()
            c = 1
            while (c < l):
                line = f.readline().split()
                c += 1
                atom1 = int(line[0])
                atom2 = int(line[1])
                reactions.append([atom1, atom2])

        for i in range(self.N):
            self.AM.append(np.zeros(self.N))
        for edge in reactions:
            self.AM[edge[0]][edge[1]] += 1
            self.AM[edge[1]][edge[0]] += 1

        return reactions

    def printAM(self):

        for i in range(len(self.AM)):
            line = ''
            for j in range(len(self.AM[i])):
                line += (str(self.AM[i][j]) + ',')

    def radicals_pdbs(self, index):

        path = self.chdir + '/'
        os.chdir(path)
        index = str(index).zfill(2)
        file_name = self.run_folder_rootname + "_" + (index) + ".out"
        file = path + file_name

        f = open(file, 'r')
        l = self.file_len(file)

        c = 0
        C = 0
        Nradicals = self.radicals0
        Npdbs = self.pdbs0
        Radicals = [Nradicals]
        PDBs = [Npdbs]
        while (c < l):
            line = f.readline().split()
            c += 1
            radicals = 0
            pdbs = 0
            run = False
            if (len(line) > 0):
                if (line[0] == "Free" and line[1] == "Radical" and line[2] == "Reactions"):
                    run = True
            while (run):
                line = f.readline().split()
                c += 1
                if (len(line) > 0):
                    if (line[-1] == "addition"):
                        Npdbs -= 1
                    if (line[-1] == "disproportionation" or line[-1] == "recombination"):
                        Nradicals -= 2
                    Radicals.append(Nradicals)
                    PDBs.append(Npdbs)
                else:
                    run = False

        return Radicals, PDBs

    def simulation_time(self, outfile_path):

        Tsim = 0
        timestep = 0.5
        f = open(outfile_path, 'r')
        l = self.file_len(outfile_path)

        c = 0
        while (c < l):
            line = f.readline().split()
            c += 1
            if (len(line) == 5):
                if (line[0] == 'NPT'):
                    Tsim += int(line[-2])

        return Tsim * timestep

    def radicals_pdbs_properties(self):

        Radicals = []
        PDBs = []

        for i in range(self.n_runs):
            radicals, pdbs = self.radicals_pdbs(i)
            Radicals.append(radicals)
            PDBs.append(pdbs)

        radicals_num_all_l = []
        for i in range(len(Radicals)):
            radicals_num_all_l.append(len(Radicals[i]))

        pdbs_num_all_l = []
        for i in range(len(PDBs)):
            pdbs_num_all_l.append(len(PDBs[i]))

        Radicals_ave = []
        for i in range(max(radicals_num_all_l)):
            Radicals_ave.append(0.0)
            s = 0
            for j in range(len(Radicals)):
                if (i < radicals_num_all_l[j]):
                    s += 1
                    Radicals_ave[-1] += Radicals[j][i]
            Radicals_ave[-1] /= s
            if ((s) / (len(Radicals)) < 1. / 3):
                del Radicals_ave[-1]

        PDBs_ave = []
        for i in range(max(pdbs_num_all_l)):
            PDBs_ave.append(0.0)
            s = 0
            for j in range(len(PDBs)):
                if (i < pdbs_num_all_l[j]):
                    s += 1
                    PDBs_ave[-1] += PDBs[j][i]
            PDBs_ave[-1] /= s
            if ((s) / (len(PDBs)) < 1. / 3):
                del PDBs_ave[-1]

        return Radicals, PDBs, Radicals_ave, PDBs_ave

    def conversion(self, index):

        timestep = 0.5

        path = self.chdir + '/'
        os.chdir(path)
        index = str(index).zfill(2)
        file_name = self.run_folder_rootname + "_" + (index) + ".out"
        file = path + file_name

        f = open(file, 'r')
        l = self.file_len(file)

        c = 0
        C = 0
        T = 0
        Time = [0]
        Conv = []
        while (c < l):
            line = f.readline().split()
            c += 1
            if (len(line) > 1):
                if (line[1] == "Atom"):
                    C += 1
            if (len(line) > 1):
                if (line[1] == "dynamics"):
                    Conv.append(C)
                    T += int(line[-2])
                    Time.append(T * timestep)

        del Time[-1]
        return Time, Conv

    def conversion_properties(self):

        C = []
        T = []

        interpolation_nodes = 100
        for i in range(self.n_runs):
            t, c = self.conversion(i)
            t_new = np.arange(t[0], t[-1], interpolation_nodes)
            f = scipy.interpolate.interp1d(t, c)
            c_new = f(t_new)
            C.append(c_new)
            T.append(t_new)

        num_all_l = []
        for i in range(len(C)):
            num_all_l.append(len(C[i]))

        C_ave = []
        for i in range(max(num_all_l)):
            C_ave.append(0.0)
            s = 0
            for j in range(len(C)):
                if (i < num_all_l[j]):
                    s += 1
                    C_ave[-1] += C[j][i]
            C_ave[-1] /= s
            if ((s) / (len(C)) < 1. / 3):
                del C_ave[-1]

        return C_ave, np.arange(0, interpolation_nodes * len(C_ave), interpolation_nodes) / 1e6

    def moment(self, list, order):

        a = 0.7  # flexible polymers: 0.7,  inflexible: 1.1

        s = 0.
        for i in list:
            s += i ** 2

        s2 = 0.
        for i in list:
            s2 += i ** 3

        v = 0.
        for i in list:
            v += i ** (1. + a)

        if (order == 0):
            return len(list)

        if (order == 1):
            return sum(list) / len(list)

        if (order == 2):
            return s / sum(list)

        if (order == 3):
            return s2 / s

        if (order == 'v' or order == 'visc' or order == 'viscosity'):
            return (v / sum(list)) ** (1 / a)

    def size_distribution(self, index):

        if (self.simulation_type == "MD"):
            path = self.chdir + '/'
            os.chdir(path)
            index = str(index).zfill(2)
            file_name = self.run_folder_rootname + "_" + (index) + ".out"
            file = path + file_name
            reactions = self.load_reactions(file)

        if (self.simulation_type == "graph"):
            os.chdir(self.chdir)
            index = str(index).zfill(2)
            file = self.chdir + str(self.run_folder_rootname) + "_" + index + ".out"
            reactions = self.load_reactions(file)

        G = nx.Graph()
        G.add_nodes_from(range(self.N))

        distr = []
        for i in range(len(reactions)):
            G.add_edge(reactions[i][0], reactions[i][1])

            size_list = []
            G_cc = nx.connected_components(G)
            for cc in G_cc:
                if (len(cc) > 1):
                    size_list.append(len(cc))
            size_list.sort()
            distr.append(size_list)

        return distr

    def size_distribution_properties(self):

        num_all = []
        mean_all = []
        weight_all = []
        z_all = []
        visc_all = []

        for i in range(self.n_runs):
            print('.', end='')
            sys.stdout.flush()
            distr = self.size_distribution(i)

            num = []
            mean = []
            weight = []
            z = []
            visc = []

            for j in range(len(distr)):
                num.append(self.moment(distr[j], 0))
                mean.append(self.moment(distr[j], 1))
                weight.append(self.moment(distr[j], 2))
                z.append(self.moment(distr[j], 3))
                visc.append(self.moment(distr[j], 'v'))

            num_all.append(num)
            mean_all.append(mean)
            weight_all.append(weight)
            z_all.append(z)
            visc_all.append(visc)

        num_all = np.array(num_all)
        mean_all = np.array(mean_all)
        weight_all = np.array(weight_all)
        z_all = np.array(z_all)
        visc_all = np.array(visc_all)

        num_all_l = []
        mean_all_l = []
        weight_all_l = []
        z_all_l = []
        visc_all_l = []

        for i in num_all:
            num_all_l.append(len(i))
        for i in mean_all:
            mean_all_l.append(len(i))
        for i in weight_all:
            weight_all_l.append(len(i))
        for i in z_all:
            z_all_l.append(len(i))
        for i in visc_all:
            visc_all_l.append(len(i))

        num_ave = []
        mean_ave = []
        weight_ave = []
        z_ave = []
        visc_ave = []

        for i in range(max(num_all_l)):
            num_ave.append(0.0)
            s = 0
            for j in range(len(num_all)):
                if (i < num_all_l[j]):
                    s += 1
                    num_ave[-1] += num_all[j][i]
            num_ave[-1] /= s
            if ((s) / (len(num_all)) < 1. / 4):
                del num_ave[-1]

        for i in range(max(mean_all_l)):
            mean_ave.append(0.0)
            s = 0
            for j in range(len(mean_all)):
                if (i < mean_all_l[j]):
                    s += 1
                    mean_ave[-1] += mean_all[j][i]
            mean_ave[-1] /= s
            if ((s) / (len(mean_all)) < 1. / 4):
                del mean_ave[-1]

        for i in range(max(weight_all_l)):
            weight_ave.append(0.0)
            s = 0
            for j in range(len(weight_all)):
                if (i < weight_all_l[j]):
                    s += 1
                    weight_ave[-1] += weight_all[j][i]
            weight_ave[-1] /= s
            if ((s) / (len(weight_all)) < 1. / 4):
                del weight_ave[-1]

        for i in range(max(z_all_l)):
            z_ave.append(0.0)
            s = 0
            for j in range(len(z_all)):
                if (i < z_all_l[j]):
                    s += 1
                    z_ave[-1] += z_all[j][i]
            z_ave[-1] /= s
            if ((s) / (len(z_all)) < 1. / 4):
                del z_ave[-1]

        for i in range(max(visc_all_l)):
            visc_ave.append(0.0)
            s = 0
            for j in range(len(visc_all)):
                if (i < visc_all_l[j]):
                    s += 1
                    visc_ave[-1] += visc_all[j][i]
            visc_ave[-1] /= s
            if ((s) / (len(visc_all)) < 1. / 4):
                del visc_ave[-1]

        '''
        self.plot_family(num_all, num_ave, "N", "Number of macromolecules", "macro_num")
        self.plot_family(mean_all, mean_ave, "mean macromolecule size", "Average size of macromolecules", "macro_mean")
        self.plot_family(weight_all, weight_ave, "weight average macromolecule size",
                         "Weight average size of macromolecules", "macro_weight")
        self.plot_family(z_all, z_ave, "z average macromolecule size", "Z average size of macromolecules", "macro_z")
        self.plot_family(visc_all, visc_ave, "viscosity average macromolecule size",
                         "Viscosity average size of macromolecules", "macro_visc")
        '''
        return num_all, weight_all, mean_all, z_all, visc_all, [num_ave, weight_ave, mean_ave, z_ave, visc_ave]

    def plot_family(self, data_all, data_ave, ylabel, title, abbr):
        N = self.max_reaction_number
        plt.clf()
        for i in range(len(data_all)):
            plt.plot(np.arange(1. / N, (len(data_all[i]) + 1.) / N, 1. / N), data_all[i], 1, lw=1.5,
                     color='xkcd:light blue')

        plt.plot(np.arange(1. / N, (len(data_ave) + 1.) / N, 1. / N), data_ave, 1, lw=2.5, color='xkcd:navy')
        plt.grid()
        plt.xlabel("conversion")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.savefig(abbr + ".png")

    def plot_degree_family(self, distr_ave, f, title):

        N = self.max_reaction_number

        X = []
        for i in range(len(distr_ave)):
            X.append(np.arange(1. / N, (float(len(distr_ave[i])) + 1) / N, 1. / N))

        N = len(self.loads)
        HSV_tuples = [(x * 1.0 / N, 0.8, 0.6) for x in range(N)]
        RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))

        n = len(distr_ave)
        while (n % 3 != 0):
            n += 1

        plt.clf()
        fig, axis = plt.subplots(int(n / 2), 2, sharex=True, sharey=True, gridspec_kw={'wspace': 0.0, 'hspace': 0.0})

        s = 0
        c = -1
        for i in range(len(axis)):
            for j in range(len(axis[i])):
                if (distr_ave[s][0] > 0.9):
                    c += 1
                axis[i][j].grid(axis='both', which='both')
                axis[i][j].set_xticks(np.arange(0, X[s][-1], np.round(max(X[s]) / 2, 1)))
                axis[i][j].plot(X[s], distr_ave[s], lw=2.5, color=RGB_tuples[c])
                if (j == 0):
                    axis[i][j].set_ylabel("fraction")
                s += 1

        axis[-1][int(len(axis[-1]) / 4)].set_xlabel("conversion")
        axis[0][int(len(axis[-1]) / 2)].set_title(title)

        plt.savefig(f, bbox_inches='tight')

        return X

    def degree_distribution(self, index, max_degrees):

        if (self.simulation_type == "MD"):
            path = self.chdir + "/"
            os.chdir(path)
            index = str(index).zfill(2)
            file_name = self.run_folder_rootname + "_" + (index) + ".out"
            file = path + file_name
            reactions = self.load_reactions(file)
        if (self.simulation_type == "graph"):
            os.chdir(self.chdir)
            index = str(index).zfill(2)
            file = self.chdir + str(self.run_folder_rootname) + "_" + index + ".out"
            reactions = self.load_reactions(file)

        degrees = []
        for i in range(len(self.loads)):
            degrees.append(np.zeros(max_degrees[i] + 1))
        for i in range(len(degrees)):
            degrees[i][0] += self.loads[i]

        AM = []
        for i in range(self.N):
            AM.append(np.zeros(self.N))

        distr = []
        for i in range((np.sum(max_degrees) + len(max_degrees))):
            distr.append([])

        for i in range(len(reactions)):
            degree1 = int(np.sum(AM[reactions[i][0]]))
            degree2 = int(np.sum(AM[reactions[i][1]]))
            type1 = 0
            type2 = 0
            s = 0
            while (reactions[i][0] - s >= self.loads[type1]):
                s += self.loads[type1]
                type1 += 1
            s = 0
            while (reactions[i][1] - s >= self.loads[type2]):
                s += self.loads[type2]
                type2 += 1

            if (degree1 + 2 > len(degrees[type1])):
                continue
            if (degree2 + 2 > len(degrees[type2])):
                continue

            degrees[type1][degree1] -= 1
            degrees[type1][degree1 + 1] += 1
            degrees[type2][degree2] -= 1
            degrees[type2][degree2 + 1] += 1

            AM[reactions[i][0]][reactions[i][1]] += 1
            AM[reactions[i][1]][reactions[i][0]] += 1

            for j in range(len(degrees)):
                for k in range(len(degrees[j])):
                    distr[int(j + np.sum(max_degrees[:j]) + k)].append(degrees[j][k])

        return distr

    def degree_distribution_properties(self, max_degrees, n_runs, plot=True):

        distr_all = []
        for i in range(n_runs):
            distr_all.append(self.degree_distribution(i, max_degrees))

        distr_all = np.array(distr_all, dtype=object)
        distr_all = np.swapaxes(distr_all, 0, 1)

        num_all_l = []
        for i in range(len(distr_all[0])):
            num_all_l.append(len(distr_all[0][i]))

        distr_ave = []
        norm = 0
        for i in range(len(distr_all)):
            if (distr_all[i][0][0] > 1.0):
                norm = distr_all[i][0][0] + 1
            distr_ave.append([])
            for j in range(max(num_all_l)):
                distr_ave[-1].append(0.0)
                s = 0
                for k in range(len(distr_all[i])):
                    if (j < num_all_l[k]):
                        s += 1
                        distr_ave[-1][-1] += distr_all[i][k][j]
                distr_ave[-1][-1] /= s
                distr_ave[-1][-1] /= norm

                if ((s) / (len(distr_all[i])) < 1. / 4):
                    del distr_ave[-1][-1]

        if (plot):
            X = self.plot_degree_family(distr_ave, "degree_distribution.png", "Degree distribution")
        else:
            X = []
            for i in range(len(distr_ave)):
                X.append(
                    np.arange(1. / self.max_reaction_number, (float(len(distr_ave[i])) + 1) / self.max_reaction_number,
                              1. / self.max_reaction_number))

        return [X, distr_ave]

    def smooth(self, arr, deg, max_conv):
        x1 = np.linspace(1 / len(arr), max_conv, len(arr))

        fit = np.polyfit(x1, arr, deg, cov=True)
        y = np.polyval(fit[0], x1)

        return y

    def plot_fit(self, c, N, N_smooth):

        a = 0
        b = self.funcs[0] + 1
        for i in range(len(self.funcs)):
            plt.clf()
            for j in range(a, b):
                plt.plot(c, N[j], marker='o', ls='', ms=1.4)
            for j in range(a, b):
                plt.plot(c, N_smooth[j], ls='--')
            plt.xlabel("conversion")
            plt.ylabel("fractional occurrence")
            plt.title("Distribution fits")
            plt.savefig(str(self.chdir) + '/fit' + str(i + 1) + '.png', bbox_inches='tight')

            a = self.funcs[i] + 1
            if (i < len(self.funcs) - 1):
                b = self.funcs[i + 1] + 1 + a
            else:
                b = sum(self.funcs) + len(self.funcs)

    def plot_fsse(self, c, N):

        a = 0
        b = self.funcs[0] + 1
        for j in range(len(self.funcs)):
            # 0 degree monomer
            plt.clf()
            plt.plot(c, -6 / np.gradient(np.log(N[0], c)))
            plt.show()
            plt.clf()
            fig, ax = plt.subplots(2, 1, sharex=True, gridspec_kw={'wspace': 0., 'hspace': 0.})

            for i in range(a, b):
                ax[0].plot(c, N[i], label="n = " + str(i - 3), lw=3)
            ax[0].grid()
            ax[0].legend()

            m = self.funcs[j]
            s0 = self.funcs[j]
            s_ = m
            ax[1].plot([], [])

            k = 1
            for n in range(a + 1, b):
                s_ = (s_ * N[n - 1] / N[n] - (1 / m) * np.array(np.gradient(np.log(N[n]), c)) * s0 * (1 - c))
                ax[1].plot(c, s_, lw=2.5, label="n = " + str(n - 3))
                ax[1].axhline((m - k), ls='--', color='r', lw=1.)
                k += 1
            ax[1].set_ylim([-4, 8])
            ax[1].grid()
            ax[1].legend()

            ax[0].set_title("MD simulations")
            ax[1].set_xlabel("conversion")
            ax[0].set_ylabel("distribution")
            ax[1].set_ylabel("FSSE factors")

            for a in ax:
                a.label_outer()
            fig.set_figheight(14.5)
            fig.set_figwidth(9.5)
            plt.savefig(str(self.chdir) + "/fsse" + str(j + 1) + "_100.png", bbox_inches='tight')

            a = self.funcs[j] + 1
            if (i < len(self.funcs) - 1):
                b = self.funcs[j + 1] + 1 + a
            else:
                b = sum(self.funcs) + len(self.funcs)

    def fsse(self):

        distr0 = self.degree_distribution_properties(self.funcs, self.n_runs, plot=True)
        max_conv = distr0[0][0][-1]

        distr = np.array(distr0[1])
        c = np.linspace(1 / len(distr[0]), max_conv, len(distr[0]))

        N = []
        N_prime = []
        for distr_ in distr:
            smooth_distr = self.smooth(distr_, 7, max_conv)
            N.append(smooth_distr)
            grad_smooth_distr = np.gradient(np.array(smooth_distr), c)
            N_prime.append(grad_smooth_distr)

        self.plot_fit(c, distr, N)

        N = np.array(N)

        self.plot_fsse(c, N)