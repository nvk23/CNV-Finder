import pandas as pd
import os
import numpy as np
import plotly.express as px
import random
from scipy.stats import iqr

# Supress copy warning.
pd.options.mode.chained_assignment = None


# Define function to check for a specific chromosome interval in the interval reference file
def check_interval(interval_name, interval_file='ref_files/glist_hg38_intervals.csv'):
    interval_dir = os.path.dirname(os.path.abspath(interval_file))

    # Handle alternate names for genes, e.g., PRKN and PARK2 - can expand to other common NDD genes
    if interval_name == 'PRKN':
        interval_name = 'PARK2'

    interval_df = pd.read_csv(interval_file)
    positions = interval_df[interval_df.NAME == interval_name]
    if len(positions) > 0:
        chrom = positions.CHR.values[0]
        start_pos = positions.START.values[0]
        stop_pos = positions.STOP.values[0]
    else:
        # Add extracted chromosome, start, and stop positions to custom_interval file for future reference if not found
        chrom, start_pos, stop_pos = None, None, None
        print('Interval name not found in interval reference file. Added to custom reference file with base pair positions you provided.')
        new_intervals = pd.DataFrame(
            {'NAME': chr, 'CHR': chr, 'START': start_pos, 'STOP': stop_pos})
        new_intervals.to_csv(f'{interval_dir}/custom_intervals.csv', mode='a')

    return chrom, start_pos, stop_pos

# Function to find the interval for a given position
def find_interval(df, position):
    for start, stop in zip(df['POS'], df['POS_stop']):
        if start <= position <= stop:
            return f"{start}-{stop}"
    return None


def create_overlapping_windows(data, window_size, num_intervals):
    # Finds necessary overlap to reach the end of the data w/ specified # windows
    total_data_points = len(data)
    # Prevents overlap of 0
    overlap = max(1, int((total_data_points - window_size) /
                  max(1, num_intervals - 1)))
    print(f'{num_intervals} total windows overlapping by: {overlap} base pairs')

    start, stop = [], []
    start_index = 0

    while start_index + window_size <= total_data_points:
        end_index = start_index + window_size
        start.append(data[start_index])

        # Prevent out of bounds error
        stop.append(data[min(end_index, total_data_points - 1)])
        start_index += overlap

    return start, stop


def mean_within_interval(row, col_name, df):
    interval_mask = (df['position'] >= row['START']) & (
        df['position'] <= row['STOP'])
    return df.loc[interval_mask, col_name].mean()


def iqr_within_interval(row, col_name, df):
    interval_mask = (df['position'] >= row['START']) & (
        df['position'] <= row['STOP'])
    return iqr(df.loc[interval_mask, col_name], interpolation='midpoint')


def std_within_interval(row, col_name, df):
    interval_mask = (df['position'] >= row['START']) & (
        df['position'] <= row['STOP'])
    return df.loc[interval_mask, col_name].std()


def dosage_full(row, col_name, df):
    interval_mask = (df['position'] >= row['START']) & (
        df['position'] <= row['STOP'])
    calls = sum(df.loc[interval_mask, col_name])

    # Catches potential division by zero error
    if len(df) > 0:
        dosage = calls/len(df)
    else:
        dosage = 0
    return dosage

# Placeholder for creating a training set
def create_train_set():
    pass


def create_test_set(master_key, num_samples, training_file, snp_metrics_path, out_path, study_name='all', interval_name=None):
    # Takes master key, study name for subset or uses whole GP2, and path to training set so no overlap with test set
    if master_key.endswith('.txt'):
        master = pd.read_csv(master_key, sep='\t')
    elif master_key.endswith('.csv'):
        master = pd.read_csv(master_key)
        
    if study_name.lower() != 'all':
        full_samples_list = master[master.study == study_name]
    else:
        full_samples_list = master
        
    train_df = pd.read_csv(training_file)
    train_df.columns = map(str.lower, train_df.columns)

    if "interval" in train_df.columns:
        train_interval = train_df[train_df.interval == interval_name]
        open_ids = full_samples_list[~full_samples_list.IID.isin(
            train_interval.iid)]
    else:
        open_ids = full_samples_list[~full_samples_list.IID.isin(train_df.IID)]
    k = min(len(open_ids), num_samples)
    test_filenames = random.sample(set(open_ids.IID), k=k)

    if 'label' in master.columns:
        test_set = master[['IID', 'label']][master['IID'].isin(test_filenames)]
        test_set.reset_index(drop=True, inplace=True)
    else:
        test_set = master.IID[master['IID'].isin(test_filenames)]
        test_set = pd.DataFrame(test_set).reset_index(drop=True)
        test_set.columns = ['IID']

    test_set['snp_metrics_path'] = 0
    remove = []

    for i in range(len(test_set)):
        sample = test_set.IID.iloc[i]
        # label = test_set.label.iloc[i]

        mfile1 = f'{snp_metrics_path}/{sample}'

        if os.path.isdir(mfile1):
            test_set['snp_metrics_path'].iloc[i] = mfile1
        else:
            remove.append(sample)  # remove these from test set

    # later fix to account for missing snp metrics that result in < num_samples
    test_set_final = test_set[~test_set.IID.isin(remove)]
    test_set_final.to_csv(f'{out_path}_testing_IDs.csv', index=False)
    print(f'{len(test_set_final)} of requested {num_samples} samples have necessary SNP metrics')


def make_window_df(chr, start, stop, split_interval, window_count, buffer):
    window_size = round(((stop+buffer) - (start-buffer))/split_interval)
    print(
        f'Interval of interest split {split_interval} times with window size of {window_size} base pairs')

    # Create intervals with no gaps between windows
    l = np.arange((start-buffer), (stop+buffer)+window_size, window_size)
    no_gaps = [value for value in l[1:-1] for _ in range(2)]
    no_gaps.insert(0, l[0])
    no_gaps.insert(len(no_gaps), l[-1])

    # Will aggregate SNP metrics into features within each window
    window_df = pd.DataFrame({'START': no_gaps[::2]})
    window_df['STOP'] = window_df['START'] + window_size

    # Create overlapping windows
    data_range = range((start-buffer), (stop+buffer))
    start, stop = create_overlapping_windows(
        data_range, window_size, window_count)

    # Final dataframe with windows
    window_df = pd.DataFrame({'START': start, 'STOP': stop})

    return window_df


def fill_window_df(sample_data):
    out_path, sample, snp_metrics_file, split_interval, total_windows, cnv_exists, chr, start, stop, buffer, min_gentrain, bim_file, pvar_file = sample_data
    window_df = make_window_df(
        chr, start, stop, split_interval, total_windows, buffer)
    metrics_df = pd.read_parquet(snp_metrics_file)

    # may need to run one ancestry label at a time depending on how bim is organized
    if bim_file or pvar_file:
        if os.path.isfile(bim_file):
            bim = pd.read_csv(bim_file, sep='\s+', header=None,
                              names=['chr', 'id', 'pos', 'bp', 'a1', 'a2'], usecols=['id'])
            sample_df = metrics_df.loc[(metrics_df.snpID.isin(bim.id)) & (
                metrics_df.GenTrain_Score >= min_gentrain)]
        elif os.path.isfile(pvar_file):
            pvar = pd.read_csv(pvar_file, sep='\s+', header=None,
                               names=['#CHROM', 'POS', 'ID', 'REF', 'ALT'], usecols=['ID'])
            sample_df = metrics_df.loc[(metrics_df.snpID.isin(pvar.ID)) & (
                metrics_df.GenTrain_Score >= min_gentrain)]
    elif 'GenTrain_Score' in metrics_df.columns:
        sample_df = metrics_df.loc[(metrics_df.GenTrain_Score >= min_gentrain)]

    sample_df_interval = sample_df[['snpID', 'chromosome', 'position', 'BAlleleFreq', 'LogRRatio']][(sample_df['chromosome'] == chr)
                                                                                                    & (sample_df['position'] >= (start-buffer))
                                                                                                    & (sample_df['position'] <= (stop+buffer))]

    # find predicted CNV type
    sample_df_interval['BAF_insertion'] = np.where((sample_df_interval['BAlleleFreq'].between(
        0.65, 0.85, inclusive='neither')) | (sample_df_interval['BAlleleFreq'].between(0.15, 0.35, inclusive='neither')), 1, 0)
    sample_df_interval['L2R_deletion'] = np.where(
        sample_df_interval['LogRRatio'] < -0.2, 1, 0)
    sample_df_interval['L2R_duplication'] = np.where(
        sample_df_interval['LogRRatio'] > 0.2, 1, 0)
    sample_df_interval['BAF_middle'] = np.where((sample_df_interval['BAlleleFreq'] >= 0.15) & (
        sample_df_interval['BAlleleFreq'] <= 0.85), 1, 0)

    sample_df_interval['ALT_pred'] = np.where(sample_df_interval['BAF_insertion'] == 1, '<INS>',
                                              np.where(sample_df_interval['L2R_deletion'] == 1, '<DEL>',
                                                       np.where(sample_df_interval['L2R_duplication'] == 1, '<DUP>', '')))
    sample_df_interval['CNV_call'] = np.where(sample_df_interval['ALT_pred'] == '', 0,
                                              np.where(sample_df_interval['ALT_pred'] != '', 1, ''))

    # only where variants fall into CNV ranges
    pred_cnv = sample_df_interval[sample_df_interval['CNV_call'] == '1']

    # gather features for ML model
    sample_df_interval = sample_df_interval.astype(
        {'BAlleleFreq': 'float', 'LogRRatio': 'float', 'CNV_call': 'int'})
    pred_cnv = pred_cnv.astype(
        {'BAlleleFreq': 'float', 'LogRRatio': 'float', 'CNV_call': 'int'})
    window_df['dosage_interval'] = window_df.apply(
        lambda row: mean_within_interval(row, 'CNV_call', sample_df_interval), axis=1)
    window_df['dosage_full'] = window_df.apply(
        lambda row: dosage_full(row, 'CNV_call', pred_cnv), axis=1)

    window_df['del_dosage_full'] = window_df.apply(
        lambda row: dosage_full(row, 'L2R_deletion', pred_cnv), axis=1)
    window_df['dup_dosage_full'] = window_df.apply(
        lambda row: dosage_full(row, 'L2R_duplication', pred_cnv), axis=1)
    window_df['ins_dosage_full'] = window_df.apply(
        lambda row: dosage_full(row, 'BAF_insertion', pred_cnv), axis=1)

    window_df['del_dosage_interval'] = window_df.apply(
        lambda row: mean_within_interval(row, 'L2R_deletion', pred_cnv), axis=1)
    window_df['dup_dosage_interval'] = window_df.apply(
        lambda row: mean_within_interval(row, 'L2R_duplication', pred_cnv), axis=1)
    window_df['ins_dosage_interval'] = window_df.apply(
        lambda row: mean_within_interval(row, 'BAF_insertion', pred_cnv), axis=1)

    window_df['avg_baf'] = window_df.apply(
        lambda row: mean_within_interval(row, 'BAlleleFreq', pred_cnv), axis=1)
    window_df['avg_mid_baf'] = window_df.apply(lambda row: mean_within_interval(
        row, 'BAlleleFreq', sample_df_interval[sample_df_interval['BAF_middle'] == 1]), axis=1)
    window_df['avg_lrr'] = window_df.apply(
        lambda row: mean_within_interval(row, 'LogRRatio', pred_cnv), axis=1)

    window_df['std_baf'] = window_df.apply(
        lambda row: std_within_interval(row, 'BAlleleFreq', pred_cnv), axis=1)
    window_df['std_mid_baf'] = window_df.apply(lambda row: std_within_interval(
        row, 'BAlleleFreq', sample_df_interval[sample_df_interval['BAF_middle'] == 1]), axis=1)
    window_df['std_lrr'] = window_df.apply(
        lambda row: std_within_interval(row, 'LogRRatio', pred_cnv), axis=1)

    window_df['iqr_baf'] = window_df.apply(
        lambda row: iqr_within_interval(row, 'BAlleleFreq', pred_cnv), axis=1)
    window_df['iqr_mid_baf'] = window_df.apply(lambda row: iqr_within_interval(
        row, 'BAlleleFreq', sample_df_interval[sample_df_interval['BAF_middle'] == 1]), axis=1)
    window_df['iqr_lrr'] = window_df.apply(
        lambda row: iqr_within_interval(row, 'LogRRatio', pred_cnv), axis=1)

    window_df['cnv_range_count'] = len(pred_cnv)
    window_df['IID'] = sample
    window_df['CHR'] = chr

    window_df.fillna(0, inplace=True)
    window_df['window'] = window_df.index
    window_df['CNV_exists'] = cnv_exists

    window_df.to_csv(f'{out_path}_samples_windows.csv',
                     mode='a', header=None, index=False)


def create_app_ready_file(test_set_id_path, test_set_path, test_result_path, out_path, prob_threshold=0.8):
    test_df = pd.read_csv(test_set_path)

    if 'Unnamed: 0' in test_df.columns:
        test_df.rename(columns={'Unnamed: 0': 'window'}, inplace=True)

    test_df['abs_iqr_lrr'] = abs(test_df['iqr_lrr'])
    max_iqr = test_df.groupby('IID')['abs_iqr_lrr'].max()

    test_results = pd.read_csv(test_result_path)
    label_path = pd.read_csv(test_set_id_path)

    # merge all necessary files
    results = test_results.merge(label_path, on='IID', how='inner')
    full_results = results.merge(
        test_df[['IID', 'cnv_range_count']], on='IID', how='inner').drop_duplicates()
    full_results = full_results.merge(max_iqr, on='IID', how='inner')

    # will only save samples above probability threshold
    above_probab = full_results[full_results['Pred Values'] >= prob_threshold]
    above_probab.to_csv(f'{out_path}_app_ready.csv', index=False)

    return above_probab


def generate_pred_cnvs(sample_data):
    sample, metrics, chr, start, stop, out_path, buffer, min_gentrain, bim_file, pvar_file = sample_data
    out_dir = os.path.dirname(os.path.abspath(out_path))

    sample = metrics.split('/')[-1].split('=')[-1]

    metrics_df = pd.read_parquet(metrics)

    if bim_file or pvar_file:
        if os.path.isfile(bim_file):
            bim = pd.read_csv(bim_file, sep='\s+', header=None,
                              names=['chr', 'id', 'pos', 'bp', 'a1', 'a2'], usecols=['id'])
            sample_df = metrics_df.loc[(metrics_df.snpID.isin(bim.id)) & (
                metrics_df.GenTrain_Score >= min_gentrain)]
        elif os.path.isfile(pvar_file):
            pvar = pd.read_csv(pvar_file, sep='\s+', header=None,
                               names=['#CHROM', 'POS', 'ID', 'REF', 'ALT'], usecols=['ID'])
            sample_df = metrics_df.loc[(metrics_df.snpID.isin(pvar.ID)) & (
                metrics_df.GenTrain_Score >= min_gentrain)]
    elif 'GenTrain_Score' in metrics_df.columns:
        sample_df = metrics_df.loc[(metrics_df.GenTrain_Score >= min_gentrain)]

    sample_df_interval = sample_df[['snpID', 'chromosome', 'position', 'BAlleleFreq', 'LogRRatio']][(sample_df['chromosome'] == chr)
                                                                                                    & (sample_df['position'] >= (start-buffer))
                                                                                                    & (sample_df['position'] <= (stop+buffer))]

    # find predicted CNV type
    sample_df_interval['BAF_insertion'] = np.where((sample_df_interval['BAlleleFreq'].between(
        0.65, 0.85, inclusive='neither')) | (sample_df_interval['BAlleleFreq'].between(0.15, 0.35, inclusive='neither')), 1, 0)
    sample_df_interval['L2R_deletion'] = np.where(
        sample_df_interval['LogRRatio'] < -0.2, 1, 0)
    sample_df_interval['L2R_duplication'] = np.where(
        sample_df_interval['LogRRatio'] > 0.2, 1, 0)

    sample_df_interval['ALT_pred'] = np.where(sample_df_interval['BAF_insertion'] == 1, '<INS>',
                                              np.where(sample_df_interval['L2R_deletion'] == 1, '<DEL>',
                                                       np.where(sample_df_interval['L2R_duplication'] == 1, '<DUP>', '')))
    sample_df_interval['CNV_call'] = np.where(sample_df_interval['ALT_pred'] == '', 0,
                                              np.where(sample_df_interval['ALT_pred'] != '', 1, ''))

    # more simplistic path relies on organized out_path selection
    pred_path = f'{out_dir}/pred_cnvs'
    os.makedirs(pred_path, exist_ok=True)
    sample_df_interval.to_csv(
        f'{pred_path}/{sample}_full_interval.csv', index=False)


def plot_variants(df, x_col='BAlleleFreq', y_col='LogRRatio', gtype_col='GT', title='snp plot', opacity=1, midline=False, cnvs=None, xmin=None, xmax=None):
    d3 = px.colors.qualitative.D3

    cmap = {
        'AA': d3[0],
        'AB': d3[1],
        'BA': d3[1],
        'BB': d3[2],
        'NC': d3[3]
    }

    cmap_ALT = {
        '<INS>': d3[0],
        '<DEL>': d3[1],
        '<DUP>': d3[2],
        '<None>': d3[7]
    }

    # gtypes_list = (df[gtype_col].unique())
    if not xmin and not xmin:
        xmin, xmax = df[x_col].min(), df[x_col].max()

    ymin, ymax = df[y_col].min(), df[y_col].max()
    xlim = [xmin-.1, xmax+.1]
    ylim = [ymin-.1, ymax+.1]

    lmap = {'BAlleleFreq': 'BAF', 'LogRRatio': 'LRR'}
    smap = {'Control': 'circle', 'PD': 'diamond-open-dot'}

    if gtype_col == 'ALT_pred' or gtype_col == 'ALT':
        cmap_choice = cmap_ALT
    else:
        cmap_choice = cmap

    if isinstance(cnvs, pd.DataFrame):
        fig = px.scatter(df, x=x_col, y=y_col, color=gtype_col, color_discrete_map=cmap_choice,
                         color_continuous_scale=px.colors.sequential.matter, width=650, height=497, labels=lmap, symbol_map=smap, hover_data=[gtype_col])
        fig.update_traces(opacity=opacity)
        fig.add_traces(px.scatter(cnvs, x=x_col, y=y_col, hover_data=[
                       gtype_col]).update_traces(marker_color="black").data)
    else:
        if gtype_col == None:
            fig = px.scatter(df, x=x_col, y=y_col, color=gtype_col, color_discrete_sequence=[
                             'grey'], width=650, height=497, labels=lmap, symbol_map=smap, opacity=opacity)
        else:
            fig = px.scatter(df, x=x_col, y=y_col, color=gtype_col, color_discrete_map=cmap_choice,
                             width=650, height=497, labels=lmap, symbol_map=smap, opacity=opacity)

    if midline:
        # Calculate the average y-value for each unique x-value
        unique_x = np.linspace(min(df[x_col]), max(df[x_col]), num=50)

        # Use cut to create bins and calculate average y within each bin
        df['x_bin'] = pd.cut(df[x_col], bins=unique_x)
        grouped_df = df[[x_col, 'x_bin', y_col]].groupby(
            'x_bin').mean().reset_index()

        fig.add_traces(px.line(grouped_df, x=x_col, y=y_col).update_traces(
            line=dict(color='red', width=3), name='Average Line').data)

    fig.update_xaxes(range=xlim, nticks=10, zeroline=False)
    fig.update_yaxes(range=ylim, nticks=10, zeroline=False)

    fig.update_layout(margin=dict(r=76, t=63, b=75))

    # fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1))

    fig.update_layout(legend_title_text='CNV Range Class')

    out_dict = {
        'fig': fig,
        'xlim': xlim,
        'ylim': ylim
    }

    fig.update_layout(title_text=f'<b>{title}<b>')

    return fig