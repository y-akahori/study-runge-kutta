# 参考
# https://vigne-cla.com/5-3/

""" 1自由度のばねますダンパモデル

4次ルンゲクッタ法により1自由度ばねますダンパモデルの微分方程式を解く。

4次ルンゲクッタの流れ
- 初期値(位置、速度)を与える。
- 初期値からdt後の速度と加速度を算出する。
    - 微分方程式からdt後の傾き(速度、加速度)を算出し、位置と速度を算出する。(k1)
    - k1からdt/2後の傾き(k2)、k2からdt/2後の傾き(k3)、k3からdt後の傾き(k4)を算出する。
    - k1~k4の傾きを加重平均によって傾きを算出する。(k)
        - k = (k1 + 2k2 + 2k3 + k4)/6
- 位置と速度を更新する。
    - 初期値とk(速度, 加速度)、dtの積からdt後の位置と速度を更新する。
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class OneDegreeOfFreedom:
    def __init__(self, m, c, k):
        # TODO 単位を見直す
        self.m = m  # 質量 [kg]
        self.c = c  # 減衰係数 [N・(s/m)]
        self.k = k  # ばね係数
        self.omega = (self.k / self.m) ** 0.5

        # 数値解用変数
        self.state = np.zeros(2)
        self.velocity = np.zeros(2)

        # ばね特性
        self.origin_point = np.array([0.0, 10.0]) # 原点
        self.start_point = np.array([0.0, -15.0]) # 初期値
        self.natural_length = 10.0  # 自然長
        self.max_compression = 5.0  # 最大圧縮量
        self.max_extension = 15.0   # 最大伸長量


    # 微分方程式
    def f(self, pre_state, pre_vel):

        # TODO 最大伸縮量導入
        state, vel = pre_state, pre_vel

        # TODO 自然長からの伸縮量を算出する
        # spring_length = self.natural_length - np.linalg.norm(state - self.origin_point)
        # restoring_force = -self.k*spring_length

        restoring_force = -self.k*state
        acc = (restoring_force - (self.c / self.m) * vel) / self.m  # 加速度: F/m = a (運動方程式)

        return vel, acc


    def execute(self):
        # 曲線plot用
        self.state_line = []
        self.velocity_line = []
        
        # 初期条件
        self.ymax = 10.0
        self.state[0], self.state[1] = self.start_point[0], -self.ymax
        self.velocity[0], self.velocity[1] = 0, -1e-6

        # 時間変数
        self.tmin = 0.0
        self.tmax = 20.0
        self.dt = 0.01
        self.t = self.tmin

        # アニメーション用データカット変数
        self.sb_dt = 0.1
        self.Nt = 1
        self.count = 1
        self.cat_val = int(self.sb_dt / self.dt)
        self.y_lis = [np.array([self.state[1], self.velocity[1]]).copy()]

        # 曲線plot用
        self.state_line.append(self.state[1])
        self.velocity_line.append(self.velocity[1])

        # ルンゲクッタ変数
        beta = [0.0, 1.0, 2.0, 2.0, 1.0]
        delta = [0.0, 0.0, self.dt / 2, self.dt / 2, self.dt]

        # TODO z成分を含めるため(5, 3)にする
        k_rk = np.zeros((5, 2))

        # 時間積分
        while self.t < self.tmax:
            sum_y = np.zeros(2)
            for k in [1, 2, 3, 4]:

                # 差分近似(位置, 速度の微分→速度, 加速度)
                
                #TODO xyz座標から次ステップの速度,加速度算出する
                # vel = np.sqrt(self.velocity[0]**2+self.velocity[1]**2)
                # vel = np.sqrt(self.velocity**2)
                # self.self.velocity[1] = vel
                
                pre_state = self.state[1] + delta[k] * k_rk[k - 1, 0]
                pre_vel = self.velocity[1] + delta[k] * k_rk[k - 1, 1]
                k_rk[k, 0], k_rk[k, 1] = self.f(pre_state, pre_vel)

                # TODO 変数名をxyzで含んだ意味にする
                sum_y += beta[k] * k_rk[k, :]

            # 傾きの更新
            # 更新式
                # 次の式 = 今の式 + (1/6) * (k1 + 2k2 + 2k3 + k4) * Δt
            self.state[1] += (self.dt / 6) * sum_y[0]
            self.velocity[1] += (self.dt / 6) * sum_y[1]

            # 曲線plot用
            self.state_line.append(self.state[1])
            self.velocity_line.append(self.velocity[1])

            # アニメーション用データ
            if self.count % self.cat_val == 0:
                self.y_lis.append(np.array([self.state[1], self.velocity[1]]).copy())
                self.Nt += 1
            self.count += 1
            self.t += self.dt
        self.t_line = np.linspace(self.tmin, self.tmax, len(self.state_line))


    def render(self, anim):
        
        if anim:
            fig, ax = plt.subplots()

            plt.axhline(y=self.natural_length, color='gray')
            def animate_move(i):
                plt.cla()
                plt.xlabel('x')
                plt.ylabel('y')
                plt.xlim(-1.0, 1.0)
                plt.ylim(-2 * self.ymax, 2 * self.ymax)
                # plt.ylim(-1.0, 1.0)
                # 変数
                v = self.y_lis[i][1]
                y = self.y_lis[i][0]
                x = self.velocity[0]  # x座標は1次元の配列
                a = self.f(y, v)[1]

                plt.hlines(self.natural_length, -1.0, 1.0, color='black')
                plt.plot([x, x], [y, self.natural_length], color='yellow')

                # 質点
                plt.scatter(x, y, s=1000)
                # 速度
                plt.quiver(x + 0.2, y, 0.0, v, color='green', scale=50.0, label='vel')
                # 加速度
                plt.quiver(x, y, 0.0, a, color='red', scale=50.0, label='acc')

            animate = animate_move
            kaisu = self.Nt - 1
            anim = animation.FuncAnimation(fig, animate, frames=kaisu, interval=1)
            plt.legend()
            anim.save("anime_bool.gif", writer="imagemagick")
            plt.show()
        else:
            plt.xlabel('t[s]')
            plt.plot(self.t_line, self.state_line, color='blue', label='state')
            plt.plot(self.t_line, self.velocity_line, color='red', label='velocity')
            plt.legend()
            plt.show()


if __name__ == '__main__':
    oneDOF = OneDegreeOfFreedom(m=10.0, c=10.0, k=10.0)
    oneDOF.execute()
    anim = 1
    oneDOF.render(anim)