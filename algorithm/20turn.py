import sqlite3
import pandas as pd

DB_PATH = "./data/stock_price_day_total.db"

import backtrader as bt
import backtrader.indicators as btind

class MATurn(bt.Indicator):
    _mindatas = 2

    lines = ('maturn',)
    plotinfo = dict(plot=True, plotname='maturn', subplot=False)
    plotlines = dict(
        maturn=dict(
            ls='', marker='o', markersize=3.0, color='blue', fillstyle='full'
        ))

    def once(self, start, end):
        maturn_array = self.lines.maturn.array
        maturn_array[start] = bt.NAN
        data0_array = self.data0.array
        data1_array = self.data1.array
        for i in range(start+1, end):
            maturn_array[i] = maturn_array[i-1]
            before = data0_array[i-1] < data1_array[i-1]
            after = data0_array[i] > data1_array[i]
            if before and after:
                maturn_array[i] = data1_array[i]

# Create a Stratey
class TestStrategy(bt.Strategy):
    params = (
        ('short_period', 18),('long_period', 20)
    )

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.datetime(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None

        # Add a MovingAverageSimple indicator

        long_sma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.long_period)
        short_sma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.short_period)
        # self.cross_up_value = CrossOverValue(self.short_sma, self.long_sma)

        self.maturn = MATurn(short_sma, long_sma)

        # 전날 종가가 파란점 아래면 1.0
        self.below_maturn_before = self.data.close(-1) < self.maturn.lines.maturn(-1)

        # 오늘 파란점의 값이 바뀌었고, 저가가 파란점 위면 1.0
        self.above_maturn_now = bt.And(self.data.low > self.maturn.lines.maturn, self.maturn.lines.maturn(-1) != self.maturn.lines.maturn)

        # 이게 1.0이면 이제 매수 타이밍 잡아야 함을 의미함
        self.pre_buy_signal = bt.And(self.below_maturn_before, self.above_maturn_now)

        self.pre_buy_signal_maturn = 0
        self.barup_cnt = 0
        # self.pre2_buy_signal = bt.If(self.pre_buy_signal, 1 - self.pre2_buy_signal,)
        #  bt.LinePlotterIndicator(self.below_maturn_5days, name='self.below_maturn_5days', subplot=True)
        #  bt.LinePlotterIndicator(self.above_maturn_now, name='self.above_maturn_now', subplot=True)
        # bt.LinePlotterIndicator(self.pre_buy_signal, name='pre_buy_signal', subplot=True)
        # bt.LinePlotterIndicator(self.pre2_buy_signal, name='pre2_buy_signal', subplot=True)

    def next(self):
        # Simply log the closing price of the series from the reference
        # self.log('Close, %.2f' % self.dataclose[0])

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        # self.first_highest = self.datas[0].high[int(-self.first_highest_index.lines.index[0])]

        if self.pre_buy_signal[0] == 1.0:  # 매수 타이밍(양봉) 잡기 '시작'해야하는 경우
            # 파란점 값을 저장해서 양봉 위치 비교할 때 쓸거다.
            self.pre_buy_signal_maturn = self.maturn.lines.maturn[0]
            self.barup_cnt = 0
        elif self.pre_buy_signal_maturn != 0:
            if self.data.open[0] < self.data.close[0]:
                self.barup_cnt += 1

        # 파란점이 '매수 타이밍 잡는 파란점'에서 '그렇지 않은 파란점'으로 변한 경우, 없던일로..
        if self.pre_buy_signal_maturn != self.maturn.lines.maturn[0]:
            self.pre_buy_signal_maturn = 0

        # 저점이 파란점 뚫은 경우, 없던일로..
        if self.pre_buy_signal_maturn > self.data.low:
            self.pre_buy_signal_maturn = 0



        # 지금 보유종목 있는 경우
        if self.position:
            if (self.data.close[0] - self.buyprice) / self.buyprice >= 0.1:
                self.sell()
            elif (self.data.close[0] - self.buyprice) / self.buyprice <= -0.1:
                self.sell()

        # 지금 보유종목 없이 only 현금인 상태이며, 신호 이후 첫 양봉인 경우 매수
        elif self.pre_buy_signal_maturn and self.pre_buy_signal[0] == 0:
            if self.data.open[0] < self.data.close[0] and self.barup_cnt == 1:
                self.buy()
                self.buyprice = self.data.close[0]
                self.pre_buy_signal_maturn = 0



def sqlite_to_data_feeds(db_path):
    with sqlite3.connect(db_path) as con:
        cur = con.cursor()

        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")

        inst_list = cur.fetchall()
        for i in range(len(inst_list)):
            inst_list[i] = inst_list[i][0]

        data_list = []
        for inst_name in inst_list:
            cur.execute("SELECT * FROM {}".format(inst_name))
            df = pd.DataFrame(cur.fetchall(), columns=(
                'date', 'open', 'high', 'low', 'close', 'volume'))
            df.set_index('date', inplace=True)

            if len(df[:]) < 100:
                continue
            df = df[:][-1000:]
            # df.index = pd.to_datetime(pd.Series(map(str, df.index)), format='%Y%m%d%H%M')
            df.index = pd.to_datetime(pd.Series(map(str, df.index)), format='%Y%m%d')
            data_list.append((bt.feeds.PandasData(dataname=df), inst_name))

    return data_list


def printTradeAnalysis(analyzer):
    '''
    Function to print the Technical Analysis results in a nice format.
    '''
    # Get the results we are interested in
    total_open = analyzer.total.open
    total_closed = analyzer.total.closed
    total_won = analyzer.won.total
    total_lost = analyzer.lost.total
    win_streak = analyzer.streak.won.longest
    lose_streak = analyzer.streak.lost.longest
    pnl_net = round(analyzer.pnl.net.total, 2)
    strike_rate = (total_won / total_closed) * 100
    # Designate the rows
    h1 = ['Total Open', 'Total Closed', 'Total Won', 'Total Lost']
    h2 = ['Strike Rate', 'Win Streak', 'Losing Streak', 'PnL Net']
    r1 = [total_open, total_closed, total_won, total_lost]
    r2 = [strike_rate, win_streak, lose_streak, pnl_net]
    # Check which set of headers is the longest.
    if len(h1) > len(h2):
        header_length = len(h1)
    else:
        header_length = len(h2)
    # Print the rows
    print_list = [h1, r1, h2, r2]
    row_format = "{:<15}" * (header_length + 1)
    print("Trade Analysis Results:")
    for row in print_list:
        print(row_format.format('', *row))


def printSQN(analyzer):
    sqn = round(analyzer.sqn, 2)
    print('SQN: {}'.format(sqn))


if __name__ == '__main__':

    data_list = sqlite_to_data_feeds(DB_PATH)

    total_won = 0
    total_lost = 0

    # 각 종목에 대해 반복한다
    for data in data_list:

        startcash = 10000000

        cerebro = bt.Cerebro()

        cerebro.addstrategy(TestStrategy)

        cerebro.adddata(data[0], name=data[1])

        cerebro.broker.setcash(startcash)

        # Add the analyzers we are interested in
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="ta")
        cerebro.addanalyzer(bt.analyzers.SQN, _name="sqn")

        # cerebro.addsizer(bt.sizers.FixedSize, stake=50)
        cerebro.addsizer(bt.sizers.PercentSizer, percents=50)

        cerebro.broker.setcommission(commission=0.0)

        print('instrument name = {}'.format(data[1]))

        # Print out the starting conditions
        print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

        # Run over everything
        strategies = cerebro.run()

        # Print out the final result
        print('Final Portfolio Value: %.2f\n' % cerebro.broker.getvalue())

        firstStrat = strategies[0]


        # print the analyzers
        ta_analysis = firstStrat.analyzers.ta.get_analysis()

        if 'won' in ta_analysis.keys():
            total_won += ta_analysis.won.total
            total_lost += ta_analysis.lost.total
            printTradeAnalysis(ta_analysis)
            printSQN(firstStrat.analyzers.sqn.get_analysis())
        else:
            print('no won')

        print(total_won, total_lost)

        # Plot the result
#        cerebro.plot(style='candlestick', barup='red', bardown='skyblue')
        del cerebro


    print('total_game={}\ttotal_won={}\ttotal_lost={}'.format(total_won+total_lost, total_won, total_lost))
    print('strike rate={:.2f}'.format(total_won / (total_won+total_lost)))
