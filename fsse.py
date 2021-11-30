import os
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from scipy.signal import savgol_filter


class ReadReactions:

    def __init__(self, simulation_type: str, output_dir: str, monomer_keys_file: str, reaction_names: list,
                 n_runs: int, file_root_name="run"):
        self.simulation_type = simulation_type
        self.output_dir = output_dir
        self.monomer_keys = monomer_keys_file
        self.reaction_names = reaction_names
        self.n_runs = n_runs
        self.file_root_name = file_root_name

        self.monomer_tags = []
        self.reactions = []
        self.load_reactions()


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

    def load_reactions(self):
        for n in range(self.n_runs):
            reactions = []
            outfile_path = self.output_dir + self.file_root_name + "_" + str(n).zfill(2) + ".out"
            f = open(outfile_path, 'r')
            l = self.file_len(outfile_path)

            if self.simulation_type == "DPD":
                reactions_ = np.loadtxt(outfile_path)[:, :2].astype(int)
                self.reactions.append(reactions_)

            if (self.simulation_type == "MD"):
                if (len(self.monomer_tags) == 0):
                    monomer_tags = self.load_monomer_tags(self.monomer_keys)
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
                self.reactions.append(reactions)

            if (self.simulation_type == "graph"):
                f.readline()
                c = 1
                while (c < l):
                    line = f.readline().split()
                    c += 1
                    atom1 = int(line[0])
                    atom2 = int(line[1])
                    reactions.append([atom1, atom2])
                self.reactions.append(reactions)


class FSSE:

    def __init__(self, loads: list, funcs: list, reactions: list):
        self.loads = loads
        self.N = sum(loads)
        self.n_runs = len(reactions)
        self.funcs = funcs
        self.max_reaction_number = min(np.array(loads) * np.array(self.funcs))

        self.monomer_tags = []
        self.reactions = []
        self.cyclenumber = 0
        self.AM = []
        self.appear = []

        self.reactions = reactions

    def degree_distribution_single(self, index):

        reactions = self.reactions[index]

        degrees = []
        for i in range(len(self.loads)):
            degrees.append(np.zeros(self.funcs[i] + 1))
        for i in range(len(degrees)):
            degrees[i][0] += self.loads[i]

        AM = []
        for i in range(self.N):
            AM.append(np.zeros(self.N))

        distr = []
        for i in range((np.sum(self.funcs) + len(self.funcs))):
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
                    distr[int(j + np.sum(self.funcs[:j]) + k)].append(degrees[j][k])

        return distr

    def degree_distribution(self, plot=False):

        distr_all = []
        for i in range(self.n_runs):
            distr_all.append(self.degree_distribution_single(i))


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

        x = np.arange(1. / self.max_reaction_number, (float(len(distr_ave[0])) + 1) / self.max_reaction_number,
                          1. / self.max_reaction_number)

        distr_ave = np.array(distr_ave)
        for j in range(len(distr_ave) - 1):
            i = j + 1
            if max(distr_ave[i]) >= 0.01:
                self.appear.append(np.nonzero(distr_ave[i] >= 0.05)[0][0])
            elif max(distr_ave[i]) >= 0.0:
                self.appear.append(np.nonzero(distr_ave[i] >= 0.0)[0][0])
            else:
                self.appear.append(-1)

        if plot:
            for i in range(len(distr_ave)):
                for j in range(len(distr_all[i])):
                    plt.plot(distr_all[i][j], color="tab:blue", lw=0.7)
                plt.plot(distr_ave[i] * self.loads[1], color="tab:orange", lw=2.5)
                plt.show()


        return x, distr_ave

    def lowess_ag(self, x, y, f=0.5, iter=10):
        n = len(x)
        r = int(ceil(f * n))
        h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
        w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
        w = (1 - w ** 3) ** 3
        yest = np.zeros(n)
        delta = np.ones(n)
        for iteration in range(iter):
            for i in range(n):
                weights = delta * w[:, i]
                b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
                A = np.array([[np.sum(weights), np.sum(weights * x)],
                              [np.sum(weights * x), np.sum(weights * x * x)]])
                beta = linalg.solve(A, b)
                yest[i] = beta[0] + beta[1] * x[i]

            residuals = y - yest
            s = np.median(np.abs(residuals))
            delta = np.clip(residuals / (6.0 * s), -1, 1)
            delta = (1 - delta ** 2) ** 2

        return yest

    def norm(self, monomer_type_idx, conv_cutoff, smooth_factors=(0.3, 0.3), plot=False):

        c, distr = self.degree_distribution()

        S0 = self.funcs[monomer_type_idx]
        idx = sum(self.funcs[:monomer_type_idx])
        if idx == 0:
            idx -= 1

        grd = self.lowess_ag(c, np.gradient(-distr[idx + 1], c), f=smooth_factors[0])
        raw_norm = S0 * distr[idx + 1] / grd
        idx = np.nonzero(c <= conv_cutoff)[0][-1]
        norm_ = raw_norm[:idx]
        norm_ = np.append(norm_, 0)
        c_ = np.append(c[:idx], 1)
        norm_factor = self.lowess_ag(c_, norm_, f=smooth_factors[1])
        norm_factor = np.interp(c, c_, norm_factor)

        if plot:
            fig, ax = plt.subplots()
            ax.plot(c, raw_norm, label='raw_norm')
            ax.plot(c, (1 - c), label='ideal')
            ax.grid()
            ax.plot(c, norm_factor, ls='--', color='xkcd:pink', label='fit_norm')
            ax.legend(frameon=False)
            ax.set_title("FSSE Normalising Factor")
            ax.set_xlabel("conversion")
            ax.set_ylabel("norm factor")
            ax.set_ylim([0., 1.2])
            for spine in ["top", "bottom", "left", "right"]:
                ax.spines[spine].set_visible(False)
            plt.show()

        return norm_factor, raw_norm

    def fsse(self, monomer_type_idx, smooth_method, smooth_factors, norm, savgol_deg=(), plot=False):

        c, distr = self.degree_distribution()
        offset = sum(self.funcs[:monomer_type_idx]) + 1
        if offset == 1:
            offset -= 1

        reacted_fraction = []
        for i in range(self.funcs[monomer_type_idx]):
            reacted_fraction.append(np.sum(distr[offset:i + offset + 1], axis=0))

        fsse = []
        c_list = []
        for i in range(self.funcs[monomer_type_idx]):
            idx = self.appear[i + offset]
            fraction = reacted_fraction[i][idx:]
            reaction_rate_ = -np.gradient(fraction, c[idx:])
            reaction_rate = []
            if i == 0:
                reaction_rate = self.lowess_ag(c[idx:], reaction_rate_, f=smooth_factors[i]) * norm[idx:]
            elif smooth_method == "lowess" or smooth_method == "loess":
                reaction_rate = self.lowess_ag(c[idx:], reaction_rate_ * norm[idx:], f=smooth_factors[i])
            elif smooth_method == "savgol" or smooth_method == "savitzky-golay":
                reaction_rate = savgol_filter(reaction_rate_ * norm[idx:], smooth_factors[i], savgol_deg[i])

            if plot:
                plt.plot(c[idx:], reaction_rate_, marker='o', markersize=1, ls='', label='rate')
                plt.plot(c[idx:], reaction_rate, label='rate fitted')
                plt.plot(c[idx:], distr[i + offset][idx:], label='distr')
                plt.legend()
                plt.show()

            fsse.append(reaction_rate / distr[i + offset][idx:])
            c_list.append(c[idx:])

        return c_list, fsse
