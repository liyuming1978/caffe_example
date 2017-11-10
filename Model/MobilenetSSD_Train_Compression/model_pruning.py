#!/usr/bin/env python
'''
you must change the first data layer as below!!!
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param {
    shape {
      dim: 1
      dim: 3
      dim: 300
      dim: 300
    }
  }
}
'''
#only for mobilenet , others if has pool inside conv, it will fail

import os,sys
caffe_python = os.path.dirname(os.path.realpath(__file__)) + '/../../python'
sys.path.insert(0, caffe_python)
os.environ["GLOG_minloglevel"] = "3"
import caffe
from caffe.proto import caffe_pb2
import numpy as np
#import utils as ut
import csv,math
import subprocess, sys
import string
import copy
import argparse
from google.protobuf import text_format
from pdb import set_trace
         
def pruning_layers(in_model, in_net, args):
    if(args.inmodel2!='null'):
        in_model2 = load_model(args.inmodel2)
    else:
        in_model2 = None
    total_layers = len(in_model.layer)
    result=[]
    result.append(np.ones(3, dtype=np.int))
    resultname=[]
    resultname.append('Input')
    for prm in in_net.params:
        k = find_layerindex_by_name(in_model, prm)
        if (in_model.layer[k].type == 'Convolution' and in_model.layer[k].convolution_param.group==1):
            try:
                if(k+2>total_layers):
                    continue
                hasBatchNorm = in_model.layer[k+1].type == 'BatchNorm'
                hasScale = in_model.layer[k+2].type == 'Scale'
                if((not hasBatchNorm) or (not hasScale)):
                    continue
                prm_1 = in_model.layer[k+1].name
                prm_2 = in_model.layer[k+2].name

                if (k+6<total_layers and
                    (in_model.layer[k+4].type == 'Convolution' or in_model.layer[k+4].type == 'DepthwiseConvolution') and
                    in_model.layer[k+4].convolution_param.group >1):
                    prmdw=in_model.layer[k+4].name
                    prmdw_1=in_model.layer[k+5].name
                    prmdw_2=in_model.layer[k+6].name
                    if (in_model.layer[k+5].type != 'BatchNorm' or in_model.layer[k+6].type != 'Scale' ):
                        raise ValueError("DW layer has no BatchNorm and Scale")
                else:
                    prmdw=''
                    prmdw_1=''
                    prmdw_2=''
                                   
                (m,n,kj,ki)=np.shape(in_net.params[prm][0].data)
                conv_flat = in_net.params[prm][0].data.flatten()
                totalval=0.0
                for i in range(np.shape(conv_flat)[0]):
                   totalval+=abs(conv_flat[i])
                cut_threadhold = totalval*args.radio/100.0/np.shape(conv_flat)[0]                    
                    
                cutdw_threadhold = 0
                if (len(prmdw)>0):
                    convdw_flat = in_net.params[prmdw][0].data.flatten()
                    totalvaldw=0.0
                    for i in range(np.shape(convdw_flat)[0]):
                       totalvaldw+=abs(convdw_flat[i])
                    cutdw_threadhold = totalvaldw*args.radio/100.0/np.shape(convdw_flat)[0]
                                        
                index = 0
                resultindex = np.zeros(m, dtype=np.int)
                for mi in range(m):
                    bcut=False
                    if (len(prmdw)>0):
                        bcut = True
                        convdw_c_flat = in_net.params[prmdw][0].data[mi].flatten()
                        for j in convdw_c_flat:
                            if abs(j)>cutdw_threadhold:  #here check dw , if dw ok, cut too
                                bcut=False
                                break
                    if not bcut:
                        bcut = True
                        conv_c_flat = in_net.params[prm][0].data[mi].flatten()
                        for j in conv_c_flat:
                            if abs(j)>cut_threadhold:  #here check dw , if dw ok, cut too
                                bcut=False
                                break
                    if not bcut:
                        index+=1
                        resultindex[mi]=1 ## if not set, it's 0 

                result.append(resultindex)
                resultname.append(prm)
                print(prm+'('+str(m)+','+str(n)+','+str(kj)+','+str(ki)+'):',cut_threadhold,cutdw_threadhold,index)                
                #cut finish - set new num_output to model
                in_model.layer[k].convolution_param.num_output=index
                if (len(prmdw)>0):
                    in_model.layer[k+4].convolution_param.num_output=index
                    in_model.layer[k+4].convolution_param.group=index
                if in_model2!=None:
                    in_model2.layer[k].convolution_param.num_output=index
                    if (len(prmdw)>0):
                        in_model2.layer[k+4].convolution_param.num_output=index
                        in_model2.layer[k+4].convolution_param.group=index
                        
            except KeyError:
                print('KeyError:'+prm)
                continue
        #else:
        #    continue
        
    save_model(in_model, args.outmodel)
    if in_model2!=None:
        save_model(in_model2, args.outmodel2)
    return result,resultname

def pruning_weights(result,resultname,in_model,in_net,args):
    #since inmodel and outmodel has the same layers, so we can reuse it
    out_net = caffe.Net(args.outmodel ,caffe.TEST)
    #in_net = caffe.Net(args.inmodel, args.inweights ,caffe.TEST)
    total_layers = len(in_model.layer)
    index=1 #the first is (3,) with all 1
    layerflag = np.zeros(total_layers, dtype=np.int)
    for prm in out_net.params:
        k = find_layerindex_by_name(in_model, prm)
        if (in_model.layer[k].type == 'Convolution' and in_model.layer[k].convolution_param.group==1):
            try:
                if(k+2<total_layers):
                    hasBatchNorm = in_model.layer[k+1].type == 'BatchNorm'
                    hasScale = in_model.layer[k+2].type == 'Scale'
                else:
                    hasBatchNorm=False
                    hasScale=False
                if(k+2>total_layers) or ((not hasBatchNorm) or (not hasScale)):
                    bottomname = in_model.layer[k].bottom[0]
                    prek = find_layerindex_by_name(in_model, bottomname)
                    #for kkk in range(len(in_model.layer[k].bottom)):
                    #    print len(in_model.layer[k].bottom[kkk]),in_model.layer[k].bottom[kkk]
                    if(in_model.layer[prek].type == 'Convolution'):                    
                        preindex = find_resultindex_by_name(resultname,bottomname)
                    else:
                        preindex = index-1
                    print (prm+': From',np.shape(in_net.params[prm][0].data),'To',np.shape(out_net.params[prm][0].data))   
                    #copy conv param, conv {m,prem,ki,kj}
                    (m,n,kj,ki)=np.shape(out_net.params[prm][0].data)
                    for mi in range(m):
                        cutni=0
                        for ni in range(n):
                            if (result[preindex][ni]==1):
                                out_net.params[prm][0].data[mi][cutni][:] = in_net.params[prm][0].data[mi][ni][:]
                                cutni+=1
                    if(len(out_net.params[prm])>1):
                        out_net.params[prm][1].data[...]=np.copy(in_net.params[prm][1].data[...])                    
                    continue
                prm_1 = in_model.layer[k+1].name
                prm_2 = in_model.layer[k+2].name
                layerflag[k]=layerflag[k+1]=layerflag[k+2]=1
                
                if (k+6<total_layers and
                    (in_model.layer[k+4].type == 'Convolution' or in_model.layer[k+4].type == 'DepthwiseConvolution') and
                    in_model.layer[k+4].convolution_param.group >1):
                    prmdw=in_model.layer[k+4].name
                    prmdw_1=in_model.layer[k+5].name
                    prmdw_2=in_model.layer[k+6].name
                    layerflag[k+4]=layerflag[k+5]=layerflag[k+6]=1
                else:
                    prmdw=''
                    prmdw_1=''
                    prmdw_2=''
                    
                (m,n,kj,ki)=np.shape(out_net.params[prm][0].data)
                print (prm+': From',np.shape(in_net.params[prm][0].data),'To',np.shape(out_net.params[prm][0].data))           
                #copy conv param, conv {m,prem,ki,kj}
                cutmi=0
                for mi in range(m):
                    cutni=0
                    if (result[index][mi]==1):
                        for ni in range(n):
                            if (result[index-1][ni]==1):
                                out_net.params[prm][0].data[cutmi][cutni][:] = in_net.params[prm][0].data[mi][ni][:]
                                cutni+=1
                        cutmi+=1
                
                #conv {m,prem,ki,kj}  convdw{m,1,ki,kj}
                #no bias 
                #bn {m,m,1} scale {m,m}
                #bndw {m,m,1} scaledw {m,m}
                out_net.params[prm_1][2].data[...] = in_net.params[prm_1][2].data[...]   #bn2
                if(len(prmdw)>0):
                    out_net.params[prmdw_1][2].data[...] = in_net.params[prmdw_1][2].data[...]   #bndw2
                    print (str(prmdw)+': From',np.shape(in_net.params[prmdw][0].data),'To',np.shape(out_net.params[prmdw][0].data))         
                cutmi=0
                for mi in range(m):
                    if(result[index][mi]==1):
                        if(len(out_net.params[prm])>1):
                            out_net.params[prm][1].data[cutmi] = in_net.params[prm][1].data[mi]  #bias
                        out_net.params[prm_1][0].data[cutmi] = in_net.params[prm_1][0].data[mi]   #bn
                        out_net.params[prm_1][1].data[cutmi] = in_net.params[prm_1][1].data[mi]   #bn1
                        out_net.params[prm_2][0].data[cutmi] = in_net.params[prm_2][0].data[mi]  #scale
                        out_net.params[prm_2][1].data[cutmi] = in_net.params[prm_2][1].data[mi]  #scale , can not out_net.params[prm_2][-1].data[cutmi]
                            
                        if(len(prmdw)>0):
                            out_net.params[prmdw][0].data[cutmi][:] = in_net.params[prmdw][0].data[mi][:]
                            if(len(out_net.params[prmdw])>1):
                                out_net.params[prmdw][1].data[cutmi] = in_net.params[prmdw][1].data[mi]  #biasdw
                            out_net.params[prmdw_1][0].data[cutmi] = in_net.params[prmdw_1][0].data[mi]   #bndw
                            out_net.params[prmdw_1][1].data[cutmi] = in_net.params[prmdw_1][1].data[mi]   #bndw1
                            out_net.params[prmdw_2][0].data[cutmi] = in_net.params[prmdw_2][0].data[mi]  #scaledw
                            out_net.params[prmdw_2][1].data[cutmi] = in_net.params[prmdw_2][1].data[mi]  #scaledw                           

                        cutmi+=1                   
                index+=1
                    
            except KeyError:
                print('KeyError:'+prm)
                continue
        else:
            if(layerflag[k]==0):
                for i in range(0,len(in_net.params[prm])):
                    out_net.params[prm][i].data[...]=np.copy(in_net.params[prm][i].data[...])
    out_net.save(args.outweights);

def find_resultindex_by_name(resultname, layer_name):
    k = 0
    while resultname[k] != layer_name:
        k += 1
        if (k >= len(resultname)):
            raise IOError('layer with name %s not found' % layer_name)
    return k

def find_layerindex_by_name(model, layer_name):
    k = 0
    while model.layer[k].name != layer_name:
        k += 1
        if (k >= len(model.layer)):
            raise IOError('layer with name %s not found' % layer_name)
    return k

def set_input(in_model, out_model):
    out_model.name = in_model.name
    #For input
    for i in range(len(in_model.input)):
        out_model.input.extend([in_model.input[i]])
        if len(in_model.input_shape) < i:
            out_model.input_shape.extend([in_model.input_shape[i]])
    for i in range(len(in_model.input_dim)):
            out_model.input_dim.extend([in_model.input_dim[i]])

def load_model(filename, phase = None):
    model = caffe_pb2.NetParameter()
    input_file = open(filename, 'r')
    text_format.Merge(str(input_file.read()), model)
    input_file.close()
    if phase is None:
        return model
    else:
        out_model = caffe_pb2.NetParameter()
        set_input(model, out_model)
        for layer in model.layer:
            included = False
            if len(layer.include) == 0:
                included = True
            if len(layer.include) > 0 and len(layer.exclude) > 0:
                raise ValueError('layer ' + layer.name + ' has both include '
                               'and exclude specified.')
            for layer_phase in layer.include:
                included = included or layer_phase.phase == phase
            for layer_phase in layer.exclude:
                included = included and not layer_phase.phase == phase
            if not included:
                continue
            out_model.layer.extend([layer])
        return out_model

def save_model(model, filename):
    output_file = open(filename, 'w')
    text_format.PrintMessage(model, output_file)
    output_file.close()

def generate_pruning(args):
    in_model = load_model(args.inmodel)
    in_net = caffe.Net(args.inmodel, args.inweights ,caffe.TEST)
    print('###########Analyze pruning: conv parm: cut_threadhold, dwcut_threadhold, cutnum##########')
    (result,resultname) = pruning_layers(in_model,in_net,args)
    print('###########Cut weights: from previous to current##########')
    pruning_weights(result,resultname,in_model,in_net,args)

def define_arguments(parser):
    parser.add_argument('--inmodel', type=str,
                        default='deploy.prototxt',
                       help='input network definition (prototxt)')
    parser.add_argument('--inmodel2', type=str,
                        default='null',
                       help='input network definition (prototxt)')    
    parser.add_argument('--inweights', type=str,
                        default='deploy.caffemodel',
                       help='input network parameters (caffemodel)')
    parser.add_argument('--outmodel', type=str,
                        default='null',
                       help='output network definition (prototxt)')
    parser.add_argument('--outmodel2', type=str,
                        default='null',
                       help='output network definition (prototxt)')    
    parser.add_argument('--outweights', type=str,
                        default='null',
                       help='output network parameters (caffemodel; will be overwritten)')
    parser.add_argument('--radio', type=float,
                        default=20,
                       help='pruning radio from 1~50')
    
def check_args(args):
    if not os.path.isfile(args.inmodel):
        raise IOError(args.inmodel+' is not exist')
    if not os.path.isfile(args.inweights):
        raise IOError(args.inweights+' is not exist')
    if args.outmodel=='null':
        ext = os.path.splitext(os.path.basename(args.inmodel))
        args.outmodel = os.path.dirname(args.inmodel)+'/'+ext[0]+'_pruning'+ext[1]
    else:
        if not os.path.isdir(os.path.dirname(args.outmodel)):
            raise IOError(args.outmodel+' dir is not exist')
    if args.outmodel2=='null' and args.inmodel2!='null':
        ext = os.path.splitext(os.path.basename(args.inmodel2))
        args.outmodel2 = os.path.dirname(args.inmodel2)+'/'+ext[0]+'_pruning'+ext[1]
    else:
        if not os.path.isdir(os.path.dirname(args.outmodel)):
            raise IOError(args.outmodel+' dir is not exist')        
    if args.outweights=='null':
        ext = os.path.splitext(os.path.basename(args.inweights))
        args.outweights = os.path.dirname(args.inmodel)+'/'+ext[0]+'_pruning'+ext[1]    
    else:
        if not os.path.isdir(os.path.dirname(args.outweights)):
            raise IOError(args.outweights+' dir is not exist')            
    return args

def parse_args():
    parser = argparse.ArgumentParser(description='pruning')
    define_arguments(parser)
    args = parser.parse_args()
    args = check_args(args)
    return args

def main(argv):
    # parse args
    caffe.set_mode_cpu()
    args = parse_args()
    generate_pruning(args)

if __name__ == '__main__':
    main(sys.argv)
