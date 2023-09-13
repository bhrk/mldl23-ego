import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

path_data="./saved_data"
plot_type='fuse_models'#"sampling"
sns.set_theme()
palette = sns.color_palette('pastel')

def main():

    match(plot_type):
        case "sampling":
            plot_sampling()
        case "fuse_models":
            plot_fuse_models()
        case _:
            print("no plots")
    print("main_finsihed")

def plot_sampling():
    global path_data,palette
    df=pd.read_csv(path_data+"/"+ "Uniform_vs_Dense.csv")
    df=df.loc[(df['Frames']!=10) & (df['Frames']!=20)] #removing some 
    

    #plotting just for train split and Uniform sampling
    df_Uni=df[df['Sampling']=='Uniform']
    df_Uni_train=df_Uni[df_Uni['Split']=='train']
    fig, ax= plt.subplots()
    sns.catplot(data=df_Uni, x="Shift", y="Accuracy Avg Top1",kind='bar',hue ='Frames',col='Split',order=['D1','D2','D3'],palette=palette,ax=ax)# , col="Frames",height=4, aspect=0.6,)

    sns.barplot(data=df_Uni_train, x="Shift", y="Accuracy Avg Top1",hue ='Frames',ax=ax)# , col="Frames",height=4, aspect=0.6,)
    ax.set_title("Train Split")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),title='Frames')
    
    fig, ax_= plt.subplots()
    df_Uni_test=df_Uni[df_Uni['Split']=='test']
    sns.barplot(data=df_Uni_test, x="Shift", y="Accuracy Avg Top1",hue ='Frames',ax=ax_)# , col="Frames",height=4, aspect=0.6,)
    ax_.set_title("Test Split")
    ax_.legend(loc='center left', bbox_to_anchor=(1, 0.5),title='Frames')

    #Dense vs Uniform: 16 frames

    df_test=df[df['Split']=='test']
    df_test_16=df_test[df_test['Frames']==16]

    fig, ax_= plt.subplots()
    sns.lineplot(data=df_test_16, x="Shift", y="Accuracy Avg Top1",marker='o',hue ='Sampling',ax=ax_)# , col="Frames",height=4, aspect=0.6,)
    ax_.set_title("Test Split")
    #ax_.legend(loc='center left', bbox_to_anchor=(1, 0.5),title='Frames')
    ax_.set_ylim(0,1)

    return
def plot_fuse_models():
    path_concat='/Users/cesaraugustoseminarioyrigoyen/Documents/CORSI/DATA_SCIENCE_POLI/II_MLDL/Project/Project1A_git/mldl23-ego/Experiment_logs/Sep06_02-33-56_concat_correct/val_precision_S04-S04.txt'
    path_sum='/Users/cesaraugustoseminarioyrigoyen/Documents/CORSI/DATA_SCIENCE_POLI/II_MLDL/Project/Project1A_git/mldl23-ego/Experiment_logs/Sep06_09-08-49_sum_correct/val_precision_S04-S04.txt'
    path_avg='/Users/cesaraugustoseminarioyrigoyen/Documents/CORSI/DATA_SCIENCE_POLI/II_MLDL/Project/Project1A_git/mldl23-ego/Experiment_logs/Sep06_20-47-59_avg_correct/val_precision_S04-S04.txt'

    file= pd.read_csv(path_concat,header=None)
    values_concat =[float(e[0].split()[2].split('%')[0]) for e in file.iloc]
    iterations_concat =[int(l[0].split()[0].split('/')[0][1:]) for l in file.iloc]

    file= pd.read_csv(path_sum,header=None)
    values_sum =[float(e[0].split()[2].split('%')[0]) for e in file.iloc]
    iterations_sum =[int(l[0].split()[0].split('/')[0][1:]) for l in file.iloc]

    file= pd.read_csv(path_avg,header=None)
    values_avg =[float(e[0].split()[2].split('%')[0]) for e in file.iloc]
    iterations_avg =[int(l[0].split()[0].split('/')[0][1:]) for l in file.iloc]
    values_avg.append(values_avg[-1])
    values_avg.append(values_avg[-1])

    # df=pd.DataFrame(columns=['Iteration','Concatenation','Sum','Average'])
    # df['Iteration']=iterations_concat
    # df['Concatenation']=values_concat
    # df['Sum']=values_sum
    # df['Average']=values_avg

    df=pd.DataFrame(columns=['Iteration','Accuracy (%)','Fuse_Method'])
    df['Iteration']=iterations_concat
    df['Accuracy (%)']= values_concat
    df['Fuse_Method']= ['Concatenation' for i in range (len(iterations_concat))]
    
    df2=pd.DataFrame(columns=['Iteration','Accuracy (%)','Fuse_Method'])
    df2['Iteration']=iterations_concat
    df2['Accuracy (%)']= values_avg
    df2['Fuse_Method']= ['Average' for i in range (len(iterations_concat))]

    df3=pd.DataFrame(columns=['Iteration','Accuracy (%)','Fuse_Method'])
    df3['Iteration']=iterations_concat
    df3['Accuracy (%)']= values_sum
    df3['Fuse_Method']= ['Sum' for i in range (len(iterations_concat))]
    
    df=pd.concat([df,df2,df3])
    fig, ax_= plt.subplots()

    sns.lineplot(x='Iteration', y='Accuracy (%)', hue='Fuse_Method', data=df,ax=ax_)
    ax_.set_ylim(50, 80)

if __name__ == '__main__':
    main()