import math

import torch
import torch.nn as nn
from torch_scatter import scatter

from DataPrepareUtils import cal_msg_edge_index
from DimeModule import DimeModule
from CoulombLayer import CoulombLayer
from D3DispersionLayer import D3DispersionLayer
from PhysModule import PhysModule
from AtomToEdgeLayer import AtomToEdgeLayer
from EdgeToAtomLayer import EdgeToAtomLayer
from EmbeddingLayer import EmbeddingLayer
from MCDropout import ConcreteDropout
from BesselCalculator import bessel_expansion_raw
from BesselCalculatorFast import BesselCalculator
from utils_functions import floating_type, device, dime_edge_expansion, softplus_inverse, \
    gaussian_rbf, info_resolver, expansion_splitter, error_message, option_solver


class PhysDimeNet(nn.Module):
    """
    Combination of PhyNet and DimeNet
    For non-bonding interaction, using atom-atom interaction in PhysNet
    For bonding interaction, using directional message passing
    Final prediction is the combination of PhysNet and DimeNet
    """

    def __init__(self,
                 n_atom_embedding,
                 modules,
                 bonding_type,
                 n_feature,
                 n_output,
                 n_dime_before_residual,
                 n_dime_after_residual,
                 n_output_dense,
                 n_phys_atomic_res,
                 n_phys_interaction_res,
                 n_phys_output_res,
                 n_bi_linear,
                 nh_lambda,
                 normalize,
                 shared_normalize_param,
                 activations,
                 restrain_non_bond_pred,
                 expansion_fn,
                 uncertainty_modify,
                 coulomb_charge_correct,
                 energy_shift=None,
                 energy_scale=None,
                 debug_mode=False,
                 action="E",
                 target_names=None,
                 batch_norm=False,
                 dropout=False,
                 requires_embedding=False,
                 **kwargs):
        """
        
        :param n_atom_embedding: number of atoms to embed, usually set to 95
        :param n_feature:
        :param n_output: 
        :param n_dime_before_residual:
        :param n_dime_after_residual:
        :param n_output_dense:
        :param n_bi_linear: Dimension of bi-linear layer in DimeNet, also called N_tensor in paper
        :param nh_lambda: non-hierarchical penalty coefficient
        :param debug_mode: 
        """
        super().__init__()
        print("------unused keys-----")
        for key in kwargs:
            print("{}: {}".format(key, kwargs[key]))

        self.requires_embedding = requires_embedding
        # convert input into a dictionary
        self.target_names = target_names
        self.action = action
        self.coulomb_charge_correct = coulomb_charge_correct
        self.uncertainty_modify = uncertainty_modify
        self.expansion_fn = expansion_splitter(expansion_fn)

        self.activations = activations.split(' ')
        modules = modules.split()
        bonding_type = bonding_type.split()
        # main modules, including P (PhysNet), D (DimeNet), etc.
        self.modules = []
        self.bonding_type = []
        # post modules are either C (Coulomb) or D3 (D3 Dispersion)
        self.post_modules = []
        self.post_bonding_type = []
        for this_module, this_bonding_type in zip(modules, bonding_type):
            '''
            Separating main module and post module
            '''
            this_module = this_module.split('[')[0]
            if this_module in ['D', 'P', 'D-noOut', 'P-noOut']:
                self.modules.append(this_module)
                self.bonding_type.append(this_bonding_type)
            elif this_module in ['C', 'D3']:
                self.post_modules.append(this_module)
                self.post_bonding_type.append(this_bonding_type)
            else:
                error_message(this_module, 'module')
        self.bonding_type_keys = set(bonding_type)

        # whether calculate long range interaction or not
        self.contains_lr = False
        for _bonding_type in self.bonding_type_keys:
            if _bonding_type.find('L') >= 0:
                self.contains_lr = True

        module_bond_combination = []
        msg_bond_type = []
        for module, bond in zip(modules, bonding_type):
            module = module.split('[')[0]
            module_bond_combination.append('{}_{}'.format(module, bond))
            if module == 'D':
                msg_bond_type.append(bond)
        self.module_bond_combination = set(module_bond_combination)
        # These bonds need to be expanded into message version (i.e. 3 body interaction)
        self.msg_bond_type = set(msg_bond_type)

        self.restrain_non_bond_pred = restrain_non_bond_pred
        self.shared_normalize_param = shared_normalize_param
        self.normalize = normalize
        self.n_phys_output_res = n_phys_output_res
        self.n_phys_interaction_res = n_phys_interaction_res
        self.n_phys_atomic_res = n_phys_atomic_res
        self.debug_mode = debug_mode
        self.n_output = n_output
        self.nhlambda = nh_lambda

        # A dictionary which parses an expansion combination into detailed information
        self.expansion_info_getter = {
            combination: info_resolver(self.expansion_fn[combination])
            for combination in self.module_bond_combination
        }

        # registering necessary parameters for some expansions if needed
        for combination in self.expansion_fn.keys():
            expansion_fn_info = self.expansion_info_getter[combination]
            if expansion_fn_info['name'] == "gaussian":
                n_rbf = expansion_fn_info['n']
                feature_dist = expansion_fn_info['dist']
                feature_dist = torch.as_tensor(feature_dist).type(floating_type)
                self.register_parameter('cutoff' + combination, torch.nn.Parameter(feature_dist, False))
                # Centers are params for Gaussian RBF expansion in PhysNet
                centers = softplus_inverse(torch.linspace(1.0, math.exp(-feature_dist), n_rbf))
                centers = torch.nn.functional.softplus(centers)
                self.register_parameter('centers' + combination, torch.nn.Parameter(centers, True))

                # Widths are params for Gaussian RBF expansion in PhysNet
                widths = [softplus_inverse((0.5 / ((1.0 - torch.exp(-feature_dist)) / n_rbf)) ** 2)] * n_rbf
                widths = torch.as_tensor(widths).type(floating_type)
                widths = torch.nn.functional.softplus(widths)
                self.register_parameter('widths' + combination, torch.nn.Parameter(widths, True))
            elif expansion_fn_info['name'] == 'defaultDime':
                n_srbf = self.expansion_info_getter[combination]['n_srbf']
                n_shbf = self.expansion_info_getter[combination]['n_shbf']
                envelop_p = self.expansion_info_getter[combination]['envelop_p']
                setattr(self, f"bessel_calculator_{n_srbf}_{n_shbf}", BesselCalculator(n_srbf, n_shbf, envelop_p))

        self.base_unit_getter = {
            'D': "edge",
            'D-noOut': "edge",
            'P': "atom",
            'P-noOut': "atom",
            'C': "atom",
            'D3': "atom"
        }

        self.dist_calculator = nn.PairwiseDistance(keepdim=True)

        self.embedding_layer = EmbeddingLayer(n_atom_embedding, n_feature)

        previous_module = 'P'
        '''
        registering main modules
        '''
        for i, (current_registering_module, bonding_type) in enumerate(zip(modules, bonding_type)):
            # contents within '[]' will be considered options
            _options = option_solver(current_registering_module)
            current_registering_module = current_registering_module.split('[')[0]
            combination = current_registering_module + "_" + bonding_type
            if current_registering_module in ['D', 'D-noOut']:
                n_dime_rbf = self.expansion_info_getter[combination]['n']
                n_srbf = self.expansion_info_getter[combination]['n_srbf']
                n_shbf = self.expansion_info_getter[combination]['n_shbf']
                dim_sbf = n_srbf * (n_shbf + 1)
                this_module = DimeModule(dim_rbf=n_dime_rbf,
                                         dim_sbf=dim_sbf,
                                         dim_msg=n_feature,
                                         n_output=n_output,
                                         n_res_interaction=n_dime_before_residual,
                                         n_res_msg=n_dime_after_residual,
                                         n_dense_output=n_output_dense,
                                         dim_bi_linear=n_bi_linear,
                                         activation=self.activations[i],
                                         uncertainty_modify=uncertainty_modify)
                if self.uncertainty_modify == 'concreteDropoutModule':
                    this_module = ConcreteDropout(this_module, module_type='DimeNet')
                self.add_module('module{}'.format(i), this_module)
                if self.base_unit_getter[previous_module] == "atom":
                    self.add_module('trans_layer{}'.format(i),
                                    AtomToEdgeLayer(n_dime_rbf, n_feature, self.activations[i]))
            elif current_registering_module in ['P', 'P-noOut']:
                this_module = PhysModule(F=n_feature,
                                         K=self.expansion_info_getter[combination]['n'],
                                         n_output=n_output,
                                         n_res_atomic=n_phys_atomic_res,
                                         n_res_interaction=n_phys_interaction_res,
                                         n_res_output=n_phys_output_res,
                                         activation=self.activations[i],
                                         uncertainty_modify=uncertainty_modify,
                                         n_read_out=int(_options['n_read_out']) if 'n_read_out' in _options else 0,
                                         batch_norm=batch_norm,
                                         dropout=dropout)
                if self.uncertainty_modify == 'concreteDropoutModule':
                    this_module = ConcreteDropout(this_module, module_type='PhysNet')
                self.add_module('module{}'.format(i), this_module)
                if self.base_unit_getter[previous_module] == "edge":
                    self.add_module('trans_layer{}'.format(i), EdgeToAtomLayer())
            elif current_registering_module in ['C', 'D3']:
                pass
            else:
                error_message(current_registering_module, 'module')
            previous_module = current_registering_module

        for i, (current_registering_module, bonding_type) in enumerate(zip(self.post_modules, self.post_bonding_type)):
            if current_registering_module == 'C':
                combination = current_registering_module + "_" + bonding_type
                self.add_module('post_module{}'.format(i),
                                CoulombLayer(cutoff=self.expansion_info_getter[combination]['dist']))
            elif current_registering_module == 'D3':
                self.add_module('post_module{}'.format(i), D3DispersionLayer(s6=0.5, s8=0.2130, a1=0.0, a2=6.0519))
            else:
                error_message(current_registering_module, 'module')

        if self.normalize:
            '''
            Atom-wise shift and scale, used in PhysNet
            '''
            if shared_normalize_param:
                shift_matrix = torch.zeros(95, n_output).type(floating_type)
                scale_matrix = torch.zeros(95, n_output).type(floating_type).fill_(1.0)
                if energy_shift is not None:
                    if isinstance(energy_shift, torch.Tensor):
                        shift_matrix[:, :] = energy_shift.view(1, -1)
                    else:
                        shift_matrix[:, 0] = energy_shift
                if energy_scale is not None:
                    if isinstance(energy_scale, torch.Tensor):
                        scale_matrix[:, :] = energy_scale.view(1, -1)
                    else:
                        scale_matrix[:, 0] = energy_scale
                shift_matrix = shift_matrix / len(self.bonding_type_keys)
                self.register_parameter('scale', torch.nn.Parameter(scale_matrix, requires_grad=True))
                self.register_parameter('shift', torch.nn.Parameter(shift_matrix, requires_grad=True))
            else:
                for key in self.bonding_type_keys:
                    shift_matrix = torch.zeros(95, n_output).type(floating_type)
                    scale_matrix = torch.zeros(95, n_output).type(floating_type).fill_(1.0)
                    if energy_shift is not None:
                        shift_matrix[:, 0] = energy_shift
                    if energy_scale is not None:
                        scale_matrix[:, 0] = energy_scale
                    shift_matrix = shift_matrix / len(self.bonding_type_keys)
                    self.register_parameter('scale{}'.format(key), torch.nn.Parameter(scale_matrix, requires_grad=True))
                    self.register_parameter('shift{}'.format(key), torch.nn.Parameter(shift_matrix, requires_grad=True))

    def freeze_prev_layers(self, freeze_extra=False):
        if freeze_extra:
            # Freeze scale, shift and Gaussian RBF parameters
            for param in self.parameters():
                param.requires_grad_(False)
        for i in range(len(self.modules)):
            getattr(self, 'module{}'.format(i)).freeze_prev_layers()

    def forward(self, data):
        # torch.cuda.synchronize(device=device)
        # t0 = time.time()

        R = data.R.type(floating_type)
        Z = data.Z
        #Q = data.Q
        N = data.N
        '''
        Note: non_bond_edge_index is for non-bonding interactions
              bond_edge_index is for bonding interactions
        '''
        atom_mol_batch = data.atom_mol_batch
        edge_index_getter = {}
        for bonding_type in self.bonding_type_keys:
            # prepare edge index
            edge_index = getattr(data, bonding_type + '_edge_index', False)
            if edge_index is not False:
                edge_index_getter[bonding_type] = edge_index + getattr(data, bonding_type + '_edge_index_correct')
            else:
                edge_index_getter[bonding_type] = torch.cat(
                    [data[_type + '_edge_index'] + data[_type + '_edge_index_correct'] for _type in bonding_type],
                    dim=-1)

        msg_edge_index_getter = {}
        for bonding_type in self.msg_bond_type:
            # prepare msg edge index
            this_msg_edge_index = getattr(data, bonding_type + '_msg_edge_index', False)
            if this_msg_edge_index is not False:
                msg_edge_index_getter[bonding_type] = this_msg_edge_index + \
                                                      getattr(data, bonding_type + '_msg_edge_index_correct')
            else:
                msg_edge_index_getter[bonding_type] = cal_msg_edge_index(edge_index_getter[bonding_type]).to(device)

        # t0 = record_data('edge index prepare', t0)

        expansions = {}
        '''
        calculating expansion
        '''
        for combination in self.module_bond_combination:
            module_name = combination.split('_')[0]
            this_bond = combination.split('_')[1]
            this_expansion = self.expansion_info_getter[combination]['name']
            if module_name in ['D', 'D-noOut']:
                # DimeNet, calculate sbf and rbf
                if this_expansion == "defaultDime":
                    n_srbf = self.expansion_info_getter[combination]['n_srbf']
                    n_shbf = self.expansion_info_getter[combination]['n_shbf']
                    expansions[combination] = dime_edge_expansion(R, edge_index_getter[this_bond],
                                                                  msg_edge_index_getter[this_bond],
                                                                  self.expansion_info_getter[combination]['n'],
                                                                  self.dist_calculator,
                                                                  getattr(self, f"bessel_calculator_{n_srbf}_{n_shbf}"),
                                                                  self.expansion_info_getter[combination]['dist'])
                else:
                    raise ValueError("Double check your expansion input!")
            elif module_name in ['P', 'P-noOut']:
                # PhysNet, calculate rbf
                if this_expansion == 'bessel':
                    this_edge_index = edge_index_getter[this_bond]
                    dist_atom = self.dist_calculator(R[this_edge_index[0, :], :], R[this_edge_index[1, :], :])
                    rbf = bessel_expansion_raw(dist_atom, self.expansion_info_getter[combination]['n'],
                                               self.expansion_info_getter[combination]['dist'])
                    expansions[combination] = rbf
                elif this_expansion == 'gaussian':
                    this_edge_index = edge_index_getter[this_bond]
                    pair_dist = self.dist_calculator(R[this_edge_index[0, :], :], R[this_edge_index[1, :], :])
                    expansions[combination] = gaussian_rbf(pair_dist, getattr(self, 'centers' + combination),
                                                           getattr(self, 'widths' + combination),
                                                           getattr(self, 'cutoff' + combination))
                else:
                    error_message(this_expansion, 'expansion')
            elif (module_name == 'C') or (module_name == 'D3'):
                '''
                In this situation, we only need to calculate pair-wise distance.
                '''
                this_edge_index = edge_index_getter[this_bond]
                expansions[combination] = self.dist_calculator(R[this_edge_index[0, :], :], R[this_edge_index[1, :], :])
            else:
                # something went wrong
                error_message(module_name, 'module')

        # t0 = record_data('calculate rbf', t0)

        '''
        mji: edge embedding
        vi:  node embedding
        '''
        vi = self.embedding_layer(Z)

        # t0 = record_data('embedding layer', t0)

        separated_last_out = {key: None for key in self.bonding_type_keys}
        separated_out_sum = {key: 0. for key in self.bonding_type_keys}

        nh_loss = torch.zeros(1).type(floating_type).to(device)
        previous_module = 'P'
        previous_bonding_type = None
        mji = None
        out, regularization = 0., 0.
        '''
        Going through main modules
        '''
        for i, (module_name, bonding_type) in enumerate(zip(self.modules, self.bonding_type)):
            # this_expansion: for DimeNet, you will get rbf_ji, sbf_kji; for PhysNet, you will get rbf
            this_expansion = expansions["{}_{}".format(module_name, bonding_type)]
            _module = self._modules['module{}'.format(i)]
            if module_name in ['D', 'D-noOut']:
                if self.base_unit_getter[previous_module] == "atom":
                    mji = self._modules['trans_layer{}'.format(i)](vi, edge_index_getter[bonding_type],
                                                                   this_expansion[0])

                mji, out, regularization = _module(mji, *this_expansion, msg_edge_index_getter[bonding_type],
                                                   edge_index_getter[bonding_type])

            elif module_name in ['P', 'P-noOut']:
                if self.base_unit_getter[previous_module] == "edge":
                    vi = self._modules['trans_layer{}'.format(i)](mji, edge_index_getter[previous_bonding_type])

                vi, out, regularization = _module(vi, edge_index_getter[bonding_type], this_expansion)

            else:
                error_message(module_name, 'module')

            if module_name[-6:] == '-noOut':
                # print('this module will not output result: .{}'.format(current_module))
                pass
            else:
                nh_loss = nh_loss + regularization
                if separated_last_out[bonding_type] is not None:
                    # Calculating non-hierarchical penalty
                    out2 = out ** 2
                    last_out2 = separated_last_out[bonding_type] ** 2
                    nh_loss = nh_loss + torch.mean(out2 / (out2 + last_out2 + 1e-7)) * self.nhlambda
                separated_last_out[bonding_type] = out
                separated_out_sum[bonding_type] = separated_out_sum[bonding_type] + out
            previous_module = module_name
            previous_bonding_type = bonding_type

        # t0 = record_data('main modules', t0)

        if self.normalize:
            '''
            Atom-wise shifting and scale
            '''
            if self.shared_normalize_param:
                for key in self.bonding_type_keys:
                    separated_out_sum[key] = self.scale[Z, :] * separated_out_sum[key] + self.shift[Z, :]
            else:
                for key in self.bonding_type_keys:
                    separated_out_sum[key] = getattr(self, 'scale{}'.format(key))[Z, :] * separated_out_sum[key] + \
                                             getattr(self, 'shift{}'.format(key))[Z, :]

        atom_prop = 0.
        for key in self.bonding_type_keys:
            atom_prop = atom_prop + separated_out_sum[key]
        #print(atom_prop.shape)

        '''
        Post modules: Coulomb or D3 Dispersion layers
        '''
        for i, (module_name, bonding_type) in enumerate(zip(self.post_modules, self.post_bonding_type)):
            this_edge_index = edge_index_getter[bonding_type]
            this_expansion = expansions["{}_{}".format(module_name, bonding_type)]
            if module_name == 'C':
                if self.coulomb_charge_correct:
                    coulomb_correction = self._modules["post_module{}".format(i)](atom_prop[:, -1], this_expansion,
                                                                                  this_edge_index, q_ref=Q, N=N,
                                                                                  atom_mol_batch=atom_mol_batch)
                else:
                    print("one of the variables needed for gradient computation has been modified by an inplace"
                          " operation: need to be fixed here, probably in function cal_coulomb_E")
                    coulomb_correction = self._modules["post_module{}".format(i)](atom_prop[:, -1], this_expansion,
                                                                                  this_edge_index)
                atom_prop[:, 0] = atom_prop[:, 0] + coulomb_correction
            elif module_name == 'D3':
                d3_correction = self._modules["post_module{}".format(i)](Z, this_expansion, this_edge_index)
                atom_prop[:, 0] = atom_prop[:, 0] + d3_correction
            else:
                error_message(module_name, 'module')

        # t0 = record_data('post modules', t0)

        if self.restrain_non_bond_pred and ('N' in self.bonding_type_keys):
            # Bonding energy should be larger than non-bonding energy
            atom_prop2 = atom_prop ** 2
            non_bond_prop2 = separated_out_sum['N'] ** 2
            nh_loss = nh_loss + torch.mean(non_bond_prop2 / (atom_prop2 + 1e-7)) * self.nhlambda

        # Total prediction is the summation of bond and non-bond prediction
        mol_pred_properties = scatter(reduce='add', src=atom_prop, index=atom_mol_batch, dim=0)

        # t0 = record_data('atom prop to molecule', t0)

        Q_pred = 0.
        D_pred = 0.
        F_pred = 0.
        if self.n_output > 1:
            # the last property is considered as atomic charge prediction
            Q_pred = mol_pred_properties[:, -1]
            Q_atom = atom_prop[:, -1]
            D_atom = Q_atom.view(-1, 1) * R
            D_pred = scatter(reduce='add', src=D_atom, index=atom_mol_batch, dim=0)

        # t0 = record_data('others', t0)
        if self.debug_mode:
            if torch.abs(mol_pred_properties.detach()).max() > 1e4:
                error_message(torch.abs(mol_pred_properties.detach()).max(), 'Energy prediction')

        if self.action == "E":
            E_pred = mol_pred_properties[:, 0]
        elif self.action == "names_and_QD":
            E_pred = mol_pred_properties[:, :-1]
        else:
            E_pred = mol_pred_properties

        output = (E_pred, F_pred, Q_pred, D_pred, nh_loss)
        if self.requires_embedding:
            output = (*output, atom_prop, atom_mol_batch, Z)
        if self.action == 'nmr':
            return atom_prop, None
        else:
            return E_pred, None

