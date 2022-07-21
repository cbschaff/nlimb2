"""Some simple networks."""
from rl.modules import CatDist
from rl.networks.base import FeedForwardNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import gin
import numpy as np
import math
from nlimb.networks.transformer_base import TransformerEncoderLayer, TransformerEncoder
from nlimb.design_distributions import DesignDistribution
from typing import Type, Sequence, Optional
import nlimb.designs as designs
from utils import nest


class DesignEmbeddings(nn.Module):
    def __init__(self, dmodel, nsymbols: int, max_seq_len: int = 100):
        nn.Module.__init__(self)
        self.dmodel = dmodel
        self.nsymbols = nsymbols
        self.max_seq_len = max_seq_len

        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dmodel, 2) * (-math.log(10000.0) / dmodel))
        pe = torch.zeros(max_seq_len, dmodel)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

        self.embeddings = nn.Embedding(nsymbols+1, dmodel)


    def forward(self, obs):
        bsz, seq_len = obs['nodes'].shape
        pe = self.pe[:seq_len].unsqueeze(0).expand(bsz, -1, -1)
        emb = self.embeddings(obs['nodes'])
        if obs['edges'].shape[1] > 0:
            edge_pe = self.pe[obs['children']] + self.pe[obs['parents']]
            pe = torch.cat([pe, edge_pe], dim=1)
            emb = torch.cat([emb, self.embeddings(obs['edges'])], dim=1)
        return emb + pe


class GrammarRuleMapper(nn.Module):
    """Maps grammar symbols to expansion rules"""
    def __init__(self, grammar, demb):
        nn.Module.__init__(self)
        self.demb = demb
        self.grammar = grammar
        self.nrules = grammar.rules.nrules
        self.nsymbols = len(grammar.symbols)
        self.rule_embeddings = nn.Embedding(self.nrules, demb)
        nn.init.constant_(self.rule_embeddings.weight, 0.0)
        self.max_rules_per_symbol = max([len(self.grammar.rules.get_ids_by_symbol(sym))
                                         for sym in range(self.nsymbols)])
        rule_mapping = torch.zeros((self.nsymbols+1, self.max_rules_per_symbol),
                                        dtype=torch.long)
        rule_mask = torch.zeros((self.nsymbols+1, self.max_rules_per_symbol), dtype=torch.float)
        self.register_buffer('rule_mapping', rule_mapping)
        self.register_buffer('rule_mask', rule_mask)

        for symbol in range(1, self.nsymbols+1): #enum.auto is 1 indexed...
            rule_ids = torch.tensor(self.grammar.rules.get_ids_by_symbol(symbol), dtype=torch.long)
            self.rule_mapping[symbol, :len(rule_ids)] = rule_ids
            self.rule_mask[symbol, len(rule_ids):] = -torch.inf

    def forward(self, symbols, embeddings, token_mask):
        rule_ids = self.rule_mapping[symbols]
        rule_embeddings = self.rule_embeddings(rule_ids)
        mask = self.rule_mask[symbols].clone()
        mask[torch.logical_not(token_mask)] = -torch.inf
        logits = (embeddings.unsqueeze(2) * rule_embeddings).sum(axis=-1) + mask
        return logits, rule_ids


class ExpansionRuleDist(CatDist):
    def __init__(self, logits, rule_ids, nnodes, parents, children):
        self.bsz, self.seq_len, self.rules_per_symbol = logits.shape
        self.nnodes = nnodes
        CatDist.__init__(self, logits=logits.view(self.bsz, -1))
        self.rule_ids = rule_ids.view(self.bsz, -1)
        self.parents = parents
        self.children = children

    def sample(self, mode=False):
        rules = []
        with torch.no_grad():
            if mode:
                inds = CatDist.mode(self)
            else:
                inds = CatDist.sample(self)
            for i, (rule_id, ind) in enumerate(zip(self.rule_ids, inds)):
                rule = int(rule_id[ind])
                position = ind // self.rules_per_symbol
                if position >= self.nnodes:
                    position = position - self.nnodes
                    rules.append((rule, int(self.parents[i, position]),
                                  int(self.children[i, position])))
                else:
                    rules.append((rule, int(position)))
        return rules, inds

    def mode(self):
        return self.sample(mode=True)

    def log_prob(self, rules=None, inds=None):
        if inds is not None:
            return CatDist.log_prob(self, inds)
        elif rules is not None:
            raise NotImplementedError('Still Need to Implement This!')




@gin.configurable(module='nlimb')
class DesignTransformer(nn.Module):
    def __init__(self, grammar, nlayers=1, dmodel=256, nheads=4,
                 activation_fn=nn.ReLU):
        self.activation_fn = activation_fn
        self.nlayers = nlayers
        self.dmodel = dmodel
        self.nheads = nheads
        self.grammar = grammar
        self.nsymbols = len(grammar.symbols)
        self.nrules = grammar.rules.nrules
        nn.Module.__init__(self)

        encoder_layer = TransformerEncoderLayer(self.dmodel, self.nheads,
                                                dim_feedforward=2*self.dmodel,
                                                activation_fn=self.activation_fn,
                                                batch_first=True,
                                                norm_first=True)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=self.nlayers)

        self.embeddings = DesignEmbeddings(self.dmodel, self.nsymbols)
        self.rule_mapping = GrammarRuleMapper(grammar, dmodel)
        self.decoder = FeedForwardNet(dmodel, units=[2 * dmodel, dmodel],
                                      activation_fn=self.activation_fn, activate_last=False)


    def forward(self, x):
        nodes = x['nodes']
        symbols = torch.cat([x['nodes'], x['edges']], dim=-1)
        emb = self.embeddings(x)
        mask = torch.cat([x['node_mask'], x['edge_mask']], dim=1)
        xout = self.decoder(self.encoder(emb, key_padding_mask=torch.logical_not(mask)))
        logits, rule_ids = self.rule_mapping(symbols, xout, mask)
        return ExpansionRuleDist(logits, rule_ids, x['nodes'].shape[1], x['parents'], x['children'])


@gin.configurable(module='nlimb')
class DesignFeedForward(nn.Module):
    def __init__(self, grammar, nlayers=2, dmodel=256, activation_fn=nn.ReLU):
        self.activation_fn = activation_fn
        self.nlayers = nlayers
        self.dmodel = dmodel
        self.grammar = grammar
        self.nsymbols = len(grammar.symbols)
        self.nrules = grammar.rules.nrules
        nn.Module.__init__(self)

        self.encoder = FeedForwardNet(dmodel, units=[2*dmodel]*nlayers + [dmodel],
                                      activation_fn=activation_fn, activate_last=False)

        self.decoder = FeedForwardNet(dmodel, units=[2*dmodel, dmodel],
                                      activation_fn=activation_fn, activate_last=False)

        self.embeddings = DesignEmbeddings(self.dmodel, self.nsymbols)
        self.rule_mapping = GrammarRuleMapper(grammar, dmodel)
        self.decoder = FeedForwardNet(dmodel, units=[2 * dmodel, dmodel],
                                      activation_fn=self.activation_fn, activate_last=False)


    def forward(self, x):
        nodes = x['nodes']
        symbols = torch.cat([x['nodes'], x['edges']], dim=-1)
        emb = self.embeddings(x)
        mask = torch.cat([x['node_mask'], x['edge_mask']], dim=1)
        xin = (emb * mask.unsqueeze(-1)).sum(dim=1)
        xout = self.decoder(emb + self.encoder(xin).unsqueeze(1))
        logits, rule_ids = self.rule_mapping(symbols, xout, mask)
        return ExpansionRuleDist(logits, rule_ids, x['nodes'].shape[1], x['parents'], x['children'])

class BatchSampler():
    def __init__(self, n, device):
        self.n = n
        self.inds = torch.arange(0, n, device=device)

    def epoch(self, batch_size):
        torch.randperm(self.n, out=self.inds)

        for i in range(self.n // batch_size):
            yield self.inds[i*batch_size:(i+1)*batch_size]
        if self.n % batch_size != 0:
            yield self.inds[(self.n // batch_size):]


class SamplingTrace():
    def __init__(self):
        self.data = []
        self.inds = []
        self.ind = 0

    def add_sample(self, inds, data):
        self.inds.append(inds)
        self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        self.ind = 0
        return self

    def __next__(self):
        if self.ind >= len(self.data):
            raise StopIteration
        else:
            data = self.data[self.ind]
            inds = self.inds[self.ind]
            self.ind += 1
            return inds, data

    def batch(self, inds):
        batched_trace = SamplingTrace()
        n = self.inds[0].shape[0]
        active_inds = torch.zeros(n, dtype=torch.bool, device=self.inds[0].device)
        batch_inds = torch.zeros(n, dtype=torch.bool, device=self.inds[0].device)
        batch_inds[inds] = True
        for active, data in zip(self.inds, self.data):
            active_inds[:] = False
            active_inds[active] = True
            inds1 = active_inds[inds]
            if not torch.any(inds1):
                break
            inds2 = torch.argsort(torch.argsort(inds[inds1]))
            inds1 = inds1.nonzero()[:, 0]
            binds = batch_inds[active_inds]
            batched_data = nest.map_structure(lambda x: x[binds][inds2], data)
            batched_trace.add_sample(inds1, batched_data)

        return batched_trace

    def epoch(self, batch_size):
        for inds in BatchSampler(len(self.inds[0]), self.inds[0].device).epoch(batch_size):
            yield inds, self.batch(inds)


class BatchedDesignSampler():
    def __init__(self, model, grammar_based_design, device):
        self.design_cls = grammar_based_design
        self.model = model
        self.device = device

    def _to_torch(self, designs):
        x = [d.to_torch() for d in designs]
        node_lens = [len(d['nodes']) for d in x]
        edge_lens = [len(d['edges']) for d in x]
        bsz = len(designs)
        out = {
            'nodes': torch.zeros((bsz, max(node_lens)), dtype=torch.long, device=self.device),
            'node_order': [],
            'edges': torch.zeros((bsz, max(edge_lens)), dtype=torch.long, device=self.device),
            'parents': torch.zeros((bsz, max(edge_lens)), dtype=torch.long, device=self.device),
            'children': torch.zeros((bsz, max(edge_lens)), dtype=torch.long, device=self.device),
            'node_mask': torch.zeros((bsz, max(node_lens)), dtype=torch.bool, device=self.device),
            'edge_mask': torch.zeros((bsz, max(edge_lens)), dtype=torch.bool, device=self.device),
        }
        for i, d in enumerate(x):
            out['nodes'][i, :node_lens[i]] = d['nodes']
            out['node_order'].append(d['node_order'])
            out['node_mask'][i, :node_lens[i]] = True
            out['edges'][i, :edge_lens[i]] = d['edges']
            out['parents'][i, :edge_lens[i]] = d['parents']
            out['children'][i, :edge_lens[i]] = d['children']
            out['edge_mask'][i, :edge_lens[i]] = True
        return out

    def sample(self, n: int, mode=False):
        designs = [self.design_cls() for _ in range(n)]
        trace = SamplingTrace()
        active_designs = designs
        while True:
            with torch.no_grad():
                active_designs = [d for d in active_designs if not d.is_complete()]
                active_inds = torch.tensor([i for i, d in enumerate(designs) if not d.is_complete()],
                                           dtype=torch.long, device=self.device)
                if len(active_designs) == 0:
                    break
                x = self._to_torch(active_designs)
                dist = self.model(x)
                if mode:
                    rules, inds = dist.mode()
                else:
                    rules, inds = dist.sample()
                node_order = x['node_order']
                del x['node_order']
                trace.add_sample(active_inds, {'input': x, 'samples': inds})
                with torch.no_grad():
                    for i, (design, rule) in enumerate(zip(active_designs, rules)):
                        if len(rule) == 3:
                            design.apply(*rule)
                        else:
                            rule_id, position = rule
                            node = node_order[i][position]
                            design.apply(rule_id, node)

        return designs, trace

    def mode(self):
        return self.sample(1, mode=True)

    def log_prob(self, trace):
        log_prob = None
        for inds, data in trace:
            logp = self.model(data['input']).log_prob(inds=data['samples'])
            if log_prob is None:
                log_prob = logp
            else:
                log_prob = torch.index_add(log_prob, 0, inds, logp)

        return log_prob

    def individual_log_prob(self, trace):
        n = len(trace)
        d = len(trace.inds[0])
        log_prob = torch.zeros((d, n), device=trace.inds[0].device)
        mask = torch.zeros((d, n), dtype=torch.bool, device=trace.inds[0].device)
        for i, (inds, data) in enumerate(trace):
            log_prob[inds, i] = self.model(data['input']).log_prob(inds=data['samples'])
            mask[inds, i] = True
        return log_prob, mask

    def entropy(self, trace):
        ent = 0.0
        n = 0
        for _, data in trace:
            step_ent = self.model(data['input']).entropy()
            ent += step_ent.sum()
            n += len(step_ent)
        return ent / n

    def individual_log_prob_and_ent(self, trace):
        n = len(trace)
        d = len(trace.inds[0])
        log_prob = torch.zeros((d, n), device=trace.inds[0].device)
        mask = torch.zeros((d, n), dtype=torch.bool, device=trace.inds[0].device)
        ent = 0.0
        n = 0
        for i, (inds, data) in enumerate(trace):
            dist = self.model(data['input'])
            log_prob[inds, i] = dist.log_prob(inds=data['samples'])
            mask[inds, i] = True
            step_ent = dist.entropy()
            ent += step_ent.sum()
            n += len(step_ent)
        return log_prob, mask, ent / n


class GrammarDesignDist(DesignDistribution):
    def __init__(self, design_class: Type[designs.Design], model: nn.Module):
        DesignDistribution.__init__(self)
        self.model = model(design_class.design_grammar)
        self.design_class = design_class
        self.batch_sampler = None

    def _init_batch_sampler(self):
        device = self.model.parameters().__next__().device
        self.batch_sampler = BatchedDesignSampler(self.model, self.design_class, device)

    def sample(self, n: Optional[int] = None) -> (Sequence[designs.Design], SamplingTrace):
        if self.batch_sampler is None:
            self._init_batch_sampler()
        return self.batch_sampler.sample(n)

    def mode(self) -> designs.Design:
        if self.batch_sampler is None:
            self._init_batch_sampler()
        mode, trace = self.batch_sampler.mode()
        return mode[0], trace

    def log_prob(self, trace: SamplingTrace):
        return self.batch_sampler.log_prob(trace)

    def individual_log_prob(self, trace: SamplingTrace):
        return self.batch_sampler.individual_log_prob(trace)

    def entropy(self, trace: SamplingTrace):
        return self.batch_sampler.entropy(trace)

    def individual_log_prob_and_ent(self, trace: SamplingTrace):
        return self.batch_sampler.individual_log_prob_and_ent(trace)

    def get_log_dict(self):
        return {}


@gin.configurable(module='nlimb')
class QuadGrammarDesignDist(GrammarDesignDist):
    def __init__(self, model):
        GrammarDesignDist.__init__(self, designs.QuadrupedGrammarDesign, model)


@gin.configurable(module='nlimb')
class HexGrammarDesignDist(GrammarDesignDist):
    def __init__(self, model):
        GrammarDesignDist.__init__(self, designs.HexapodGrammarDesign, model)


@gin.configurable(module='nlimb')
class RoboGrammarDesignDist(GrammarDesignDist):
    def __init__(self, model):
        GrammarDesignDist.__init__(self, designs.RoboGrammarDesign, model)


@gin.configurable(module='nlimb')
class Grammar0509DesignDist(GrammarDesignDist):
    def __init__(self, model):
        GrammarDesignDist.__init__(self, designs.Grammar0509Design, model)


@gin.configurable(module='nlimb')
class Grammar0524DesignDist(GrammarDesignDist):
    def __init__(self, model):
        GrammarDesignDist.__init__(self, designs.Grammar0524Design, model)


@gin.configurable(module='nlimb')
class Grammar0601DesignDist(GrammarDesignDist):
    def __init__(self, model):
        GrammarDesignDist.__init__(self, designs.Grammar0601Design, model)
