import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def get_stock_data(stock_symbol, days):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)
    data = yf.download(stock_symbol, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    return data[['Open', 'High', 'Low', 'Close', 'Volume']]

def average(input_days, stock):
    total_days = input_days - 1
    print(f"Weekly Average (calendar weeks) for last {total_days} days of stock '{stock}':\n")

    data = get_stock_data(stock, total_days).reset_index()

    data['Week'] = data['Date'].dt.to_period('W').apply(lambda r: r.start_time)
    grouped = data.groupby('Week')

    weekly_data = []
    for week, group in grouped:
        avg_row = {
            'Start_Date': group['Date'].min().strftime('%Y-%m-%d'),
            'End_Date': group['Date'].max().strftime('%Y-%m-%d'),
            'Avg_Open': round(group['Open'].mean(), 2),
            'Avg_High': round(group['High'].mean(), 2),
            'Avg_Low': round(group['Low'].mean(), 2),
            'Avg_Close': round(group['Close'].mean(), 2),
            'Avg_Volume': round(group['Volume'].mean(), 2)
        }
        weekly_data.append(avg_row)

    df_weekly = pd.DataFrame(weekly_data)
    print(df_weekly)
    return df_weekly

def individual(input_days, stock):
    remaining_days = (input_days - 1) % 7
    if remaining_days == 0:
        print("No remaining days for individual data.")
        return None

    print(f"\n📄 Individual data for last {remaining_days} remaining days of stock '{stock}':\n")
    data = get_stock_data(stock, input_days - 1)
    df_individual = data.tail(remaining_days).reset_index()
    print(df_individual)
    return df_individual


stock_name = 'SBIN.NS'
days_input = 25


df_avg = average(days_input, stock_name)
df_ind = individual(days_input, stock_name)


file_name = f"{stock_name.replace('.', '_')}_stock_analysis.xlsx"

with pd.ExcelWriter(file_name) as writer:
    if df_avg is not None:
        df_avg.to_excel(writer, sheet_name='Weekly_Average', index=False)

    if df_ind is not None:
        if isinstance(df_ind.columns, pd.MultiIndex):
            df_ind.columns = ['_'.join(col).strip() for col in df_ind.columns.values]
        df_ind.to_excel(writer, sheet_name='Individual_Days', index=False)


