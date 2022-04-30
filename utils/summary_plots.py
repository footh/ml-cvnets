# redirect output of a run to a txt file and provide as input here to plot Loss, top1, top5 for test, val and val ema

import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import csv

class SummaryPlots(object):
    """
        Generate different kinds of summary plots from output from a run.
        
        Assumes : line after "*** Training summary" will have "loss=1.5650 || top1=64.6799 || top5=85.8776"
                  line after "*** Validation summary" will have "loss=2.8883 || top1=49.9630 || top5=75.2963"
                  line after "*** Validation (Ema) summary" will have "loss=2.4126 || top1=56.9259 || top5=80.8148
                  in input run output txt file
    """
    
    def __init__(self, path_to_run_output_file=None):
        self.path_to_run_output_file = path_to_run_output_file
        
        
    def input_file_exists(self):
        if os.path.exists(self.path_to_run_output_file):
            return True
        else: 
            print("Input file does not exists.")
            return False
        
    def parse_values_in_summary_line(self, next_line):
        # assumes values occur after "=" and in this order 
        # "loss=2.8883 || top1=49.9630 || top5=75.2963"
        split_parts = next_line.split("=")
        loss = float(split_parts[1].split("||")[0])
        top1 = float(split_parts[2].split("||")[0])
        top5 = float(split_parts[-1])
        
        return loss, top1, top5
    
    def parse_lr_values(self, prev_line):
        split_parts = prev_line.split("LR: [")
        lr_str = split_parts[1].split(",")[0]
        lr = float(lr_str)
        
        return lr
        
    def gen_loss_top1_top5(self):
        train_epoch_loss_top1_top5 = [] # np.zeros((lines.count, 4))
        val_epoch_loss_top1_top5 = [] # np.zeros((lines.count, 4))
        valema_epoch_loss_top1_top5 = [] # np.zeros((lines.count, 4))        
        if (self.input_file_exists()):
            lines = []
            with open(self.path_to_run_output_file) as f:
                lines = f.readlines()
                total_lines = len(lines)
                
                for line_num, line in enumerate(lines):
                    if line_num + 1 < total_lines: 
                        prev_line = lines[line_num - 1] # to get 
                        next_line = lines[line_num + 1]                        
                        if "*** Training summary" in line:
                            epoch = int(line.split(" ")[-1])
                            loss, top1, top5 = self.parse_values_in_summary_line(next_line)
                            lr = self.parse_lr_values(prev_line)
                            train_epoch_loss_top1_top5.append( (epoch, loss, top1, top5, lr) )
                        elif "*** Validation summary" in line:
                            epoch = int(line.split(" ")[-1])
                            loss, top1, top5 = self.parse_values_in_summary_line(next_line)   
                            lr = self.parse_lr_values(prev_line)
                            val_epoch_loss_top1_top5.append( (epoch, loss, top1, top5, lr) )                         
                        elif "*** Validation (Ema) summary" in line:
                            epoch = int(line.split(" ")[-1])
                            loss, top1, top5 = self.parse_values_in_summary_line(next_line)  
                            lr = self.parse_lr_values(prev_line)
                            valema_epoch_loss_top1_top5.append( (epoch, loss, top1, top5, lr) )                            
                        else: 
                            continue
         
        return np.array(train_epoch_loss_top1_top5), np.array(val_epoch_loss_top1_top5), np.array(valema_epoch_loss_top1_top5)             
        
    def plot_loss_train_val(self, train, val, mtrain=None, mval=None):      
        if mtrain is None:  
            output_path = os.path.splitext(self.path_to_run_output_file)[0] + "_loss.png"
        else: 
            output_path = os.path.splitext(self.path_to_run_output_file)[0] + "_2in1_loss.png"
        
        fig, (ax, ax2) = plt.subplots(2, 1)
        ax.plot(train[:,0], train[:,1], 'b--', label="resnet train") # epoch vs train loss
        ax.plot(val[:,0], val[:,1], '#069af3', linewidth=3, label="resnet val") # epoch vs val loss
        if mtrain is not None: 
            ax.plot(mtrain[:,0], mtrain[:,1], 'r--', label="mobileViT train")
            ax.plot(mval[:,0], mval[:,1], '#fa8072', linewidth=3, label="mobileViT val")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_ylim([0, 8.0])
        ax.set_title("training and validation loss")
        ax.legend(loc="best") # "upper center"
        
        ax2.plot(train[:,0], train[:,4], '.b', label="resnet learn rate") # epoch vs lr 
        ax2.set_ylabel("lr resnet")
        ax2.legend(loc="center right", framealpha=0.2)  
        if mtrain is not None:
            ax3 = ax2.twinx()
            ax3.plot(mtrain[:,0], mtrain[:,3], '.r', label="mobileViT learn rate") # epoch vs lr 
            ax3.set_ylabel("lr mobileViT")
            ax3.legend(loc="upper right", framealpha=0.2)
        #ax2.set_ylim([0.0, 0.5])    
                
        fig.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
    
    def plot_top1_train_val(self, train, val, mtrain=None, mval=None):
        if mtrain is None:  
            output_path = os.path.splitext(self.path_to_run_output_file)[0] + "_top1.png"
        else: 
            output_path = os.path.splitext(self.path_to_run_output_file)[0] + "_2in1_top1.png"                    
        
        fig, (ax, ax2) = plt.subplots(2, 1)
        ax.plot(train[:,0], train[:,2], 'b--', label="resnet train") # epoch vs train loss
        ax.plot(val[:,0], val[:,2], '#069af3', linewidth=3, label="resnet val") # epoch vs val loss 
        if mtrain is not None:
            ax.plot(mtrain[:,0], mtrain[:,2], 'r--', label="mobileViT train")
            ax.plot(mval[:,0], mval[:,2], '#fa8072', linewidth=3, label="mobileViT val")        
        ax.set_xlabel("epoch")
        ax.set_ylabel("top1 percent")
        ax.set_ylim([0, 100])
        ax.set_title("training and validation top1")
        ax.legend(loc="best", framealpha=0.2) # "upper center"
        
        ax2.plot(train[:,0], train[:,4], '.b', label="resnet learn rate") # epoch vs lr 
        ax2.set_ylabel("lr resnet")
        ax2.legend(loc="center right", framealpha=0.2)  
        if mtrain is not None:
            ax3 = ax2.twinx()
            ax3.plot(mtrain[:,0], mtrain[:,3], '.r', label="mobileViT learn rate") # epoch vs lr 
            ax3.set_ylabel("lr mobileViT")
            ax3.legend(loc="upper right", framealpha=0.2)
        #ax2.set_ylim([0.0, 0.5])        
        fig.savefig(output_path, format='png', dpi=300, bbox_inches='tight')            
        
    
    def plot_top5_train_val(self, train, val):
        output_path = os.path.splitext(self.path_to_run_output_file)[0] +"_top5.png"
        
        fig, ax = plt.subplots()
        ax.plot(train[:,0], train[:,3], label="train") # epoch vs train loss
        ax.plot(val[:,0], val[:,3], label="val") # epoch vs val loss 
        ax.set_xlabel("epoch")
        ax.set_ylabel("top5 percent")
        ax.set_ylim([0, 100])        
        ax.set_title("training and validation top5")
        ax.legend(loc="upper center")
        ax2 = ax.twinx()
        ax2.plot(train[:,0], train[:,4], '.g', label="learn rate") # epoch vs lr 
        ax2.set_ylabel("learning rate")
        ax2.legend(loc='lower center')
        ax2.set_ylim([0.0, 0.5])         
        fig.savefig(output_path, format='png', dpi=300, bbox_inches='tight')  
        
    def get_values_from_csv(self, csv_file, epochs=200):
        # can include : row_name1="Train/Top1", row_name2="Val/Top1", row_name3="Val_EMA/Top1"
        train_epoch_loss_top1 = []
        val_epoch_loss_top1 = []
        valema_epoch_loss_top1 = []
        # sequence : LR / Train_Loss / Val Loss / Common/BestMetric / Val_EMA_Loss 
        #                / Train_Top1 / Val_Top1 / Val_EMA_Top1     
        with open(csv_file) as f:
            reader = csv.reader(f)
            data = list(reader)            
            data = data[1:][:] # w/o header row
            for i in range(epochs):
                epoch = int(float((data[i][2]))) 
                lr    = float(data[i][1])
                train_loss  = float(data[i+200][1])
                val_loss    = float(data[i+400][1])
                valema_loss = float(data[i+800][1])
                train_top1  = float(data[i+1000][1])
                val_top1    = float(data[i+1200][1])
                valema_top1 = float(data[i+1400][1])
                train_epoch_loss_top1.append((epoch, train_loss, train_top1, lr))
                val_epoch_loss_top1.append((epoch, val_loss, val_top1, lr))
                valema_epoch_loss_top1.append((epoch, valema_loss, valema_top1, lr))
            
        return np.array(train_epoch_loss_top1), np.array(val_epoch_loss_top1), np.array(valema_epoch_loss_top1)             
                   
    def gen_plots(self, train, val, mtrain=None, mval=None):
        self.plot_loss_train_val(train, val, mtrain, mval)
        self.plot_top1_train_val(train, val, mtrain, mval)
        self.plot_top5_train_val(train, val)
        

## Test 
def main(argv):
    #input_run_file = "results\AU_results_resnet_tiny_correct_class.txt"
    #input_run_file = "results\AU_results_resnet_tiny_correct_class_depth18.txt"
    #input_run_file = "results\AU_results_resnet_tiny_correct_class_depth18_cyclic.txt"
    #input_run_file = "results\AU_results_resnet_tiny_correct_class_depth18_augment.txt"
    #input_run_file = "results\AU_results_resnet_tiny_correct_class_depth18_augment2.txt"
    #input_run_file = "results\AU_results_resnet_tiny_correct_class_depth50_augment3.txt"
    #input_run_file = "results\AU_results_resnet_tiny_correct_class_depth50_augment4_cutout.txt"
    #input_run_file = "results\AU_results_resnet_tiny_correct_class_depth50_augment4_cutout_300.txt"
    #input_run_file = "results\AU_results_resnet_tiny_correct_class_depth50_augment4_cutout_wd1.txt"
    #input_run_file = "results\AU_results_resnet_tiny_correct_class_depth50_augment4_cutout_wd2.txt"
    #input_run_file = "results\AU_results_resnet_tiny_correct_class_depth50_lr0pt1.txt"
    #input_run_file = "results\AU_results_resnet_tiny_correct_class_depth50_lr0pt1-2.txt"
    #input_run_file = "results\AU_results_resnet_tiny_correct_class_depth18_augment4_divbytenlr.txt"
    #input_run_file = "results\AU_results_resnet_tiny_correct_class_depth50_lr0pt4__divbytenlr.txt"
    #input_run_file = "results\AU_results_resnet_tiny_correct_class_depth50_augment4_divbytenlr.txt"
    #input_run_file = "results\AU_results_resnet_tiny_correct_class_depth50_augment4_divbyten_lr0pt1_3step.txt"
    #input_run_file = "results\AU_results_resnet_tiny_correct_class_depth50_augment4_divbyten_lr0pt4_3step.txt"
    #input_run_file = "results\AU_results_resnet_tiny_correct_class_depth18_augment4_divbyten_lr0pt4_3step.txt"
    #input_run_file = "results\AU_results_resnet_tiny_correct_class_depth34.txt"
    #input_run_file = "results\AU_results_resnet_tiny_correct_class_depth18_vbs1.txt"
    input_run_file = "results\AU_results_resnet_tiny_correct_class_depth18_augment4_divbyten_lr0pt4_3step_vbs1.txt"
    
    input_mobileViT_csv = None
    if len(argv) < 1: 
        input_run_file = input_run_file        
    else:
        input_run_file = argv[0]
        if len(argv) > 1:
            input_mobileViT_csv = argv[1]
            
    plots = SummaryPlots(input_run_file)
    train, val, valema = plots.gen_loss_top1_top5()  
        
    if len(train) == 0:
        print("No values to plot.")
    else:
        if input_mobileViT_csv is not None:
            mtrain, mval, mvalema = plots.get_values_from_csv(input_mobileViT_csv)   
            plots.gen_plots(train, valema, mtrain, mvalema)
        else: 
            plots.gen_plots(train, valema)

if __name__ == "__main__":
    main(sys.argv[1:])
