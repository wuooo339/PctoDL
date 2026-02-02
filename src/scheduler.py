import math
class PID:
    def __init__(self,power_cap,alpha,kp,ki,kd,index_list,log_txt):
        self.power_cap = power_cap 
        self.last_powe_cap = power_cap
        self.alpha = 100 * (1-alpha)
        self.current_power = 0
        self.error = 0     
        self.kp = kp 
        self.ki = ki 
        self.kd = kd 
        self.cur_kp = kp
        self.cur_ki = ki
        self.cur_kd = kd
        self.previous_error = 0
        self.last_error = 0
        self.times=0
        self.integral = 0
        self.index_list = index_list
        self.log_txt = log_txt
    def scheduler(self,current_power,current):
        if self.last_powe_cap!=self.power_cap:
            self.times = 0
            self.integral = 0
            self.log_txt.write("power cap change from: {:.6f}\t to: {:6f}\t \n".format(self.last_powe_cap,self.power_cap))
            self.last_error = 0
            self.previous_error = 0
        self.error = self.power_cap - current_power
        delta_t = 100*self.error / self.power_cap
        if delta_t <= self.alpha and delta_t > 2:
            self.times=0
            self.integral=0
            return current

        self.integral += self.error
        differential = self.error - 2*self.previous_error + self.last_error
        if self.error*self.previous_error<0 :
            self.integral = self.error
            self.cur_ki = 0
        elif self.times!=0:
            self.cur_ki=self.ki*math.pow(100,self.times)
        self.log_txt.write("error: {:.6f}\t power cap: {:6f}\t bili: {:.6f} times:{}\n".format(self.error,self.power_cap,delta_t,self.times))
        delta_t = abs(delta_t)
        if delta_t > 50:
            self.cur_kp = self.kp * 10
            self.integral = self.error / 4
            self.cur_ki = 0
        elif delta_t>30 and delta_t<=50:
            self.cur_kp = self.kp * 10
            self.integral = self.error /3
            self.cur_ki = 0
        elif delta_t>20 and delta_t<=30:
            self.cur_kp = self.kp * 5
            self.integral = self.error / 2
            self.cur_ki = 0
        elif delta_t>10 and delta_t<=20:
            self.cur_kp = self.kp * 10
        elif delta_t>5 and delta_t<=10:
            self.cur_kp = self.kp
        elif delta_t >=0:
            self.cur_kp = self.kp
        self.cur_kd = self.kd
        delta = self.cur_kp * self.error + self.cur_ki * self.integral + self.cur_kd * differential
        self.log_txt.write("cur_kp: {:.6f}\t cur_ki: {:6f}\t cur_kd: {:.6f}\n".format(self.cur_kp,self.cur_ki,self.cur_kd))
        self.log_txt.write("error: {:.3f}\t integral: {:3f}\t differential: {:.3f}\t delta: {:.3f}\n".format(self.error,self.integral,differential,delta))

        current_index = self.index_list.index(current)
        next_index = 0
        if current_index<=1:
            next_index = math.ceil((current_index+1) * (1 + delta))-1
        else:
            next_index = math.floor(current_index * (1 + delta))
        if next_index == current_index and self.error > 0:
            next_index = min(current_index + 1, len(self.index_list) - 1)

        self.log_txt.write("current index: {}\t next index: {}\n ".format(current_index,next_index))
        next_index = max(next_index,0)
        next_index = min(next_index,len(self.index_list)-1)
        self.log_txt.write("current : {:.3f}\t index: {}\t next index: {}\t next: {:3f}\n ".format(current,current_index,next_index,self.index_list[next_index]))
        self.last_error = self.previous_error
        self.previous_error = self.error
        self.last_powe_cap = self.power_cap
        if next_index==current_index and delta_t> 5:
            self.times+=1
        else:
            self.times=0
        return self.index_list[next_index]
 
class PIDScheduler:
    def __init__(self,sm_pid,mem_pid,bs_pid,log_txt):
        self.sm_pid = sm_pid
        self.mem_pid = mem_pid
        self.bs_pid = bs_pid
        self.log_txt = log_txt
    def scheduler(self,current_power,current_sm,current_mem,current_bs):
        self.log_txt.write("sm info\n")
        next_sm = self.sm_pid.scheduler(current_power,current_sm)
        next_mem=current_mem
        next_bs=current_bs
        return next_sm,next_mem,next_bs
class Morak:
    def __init__(self,power_cap,alpha,slo,belta,sm_clocks,batch_size_list,log_txt):
        self.power_cap = power_cap
        self.alpha = alpha
        self.belta = belta
        self.slo = slo
        self.sm_clocks = sm_clocks
        self.batch_size_list = batch_size_list
        self.log_txt = log_txt
        self.steps = {
            "huge": 10,   
            "large": 5,   
            "medium": 2,  
            "small": 1    
        }
        self.batch_step = 1

    def _get_dynamic_step(self, current_power):
        gap_ratio = abs(current_power - self.power_cap) / self.power_cap

        if gap_ratio > 0.20:    
            return self.steps["huge"]
        elif gap_ratio > 0.10: 
            return self.steps["large"]
        elif gap_ratio > 0.05:  
            return self.steps["medium"]
        else:                   
            return self.steps["small"]

    def scheduler(self,lantecy,max_power,frequency,batch_size):
        sm_index = self.sm_clocks.index(frequency)
        batch_index = self.batch_size_list.index(batch_size)
        step = self._get_dynamic_step(max_power)
        if max_power<=self.alpha*self.power_cap:
            if lantecy<=self.belta*self.slo:
                if batch_index < len(self.batch_size_list) - 1:
                    batch_index += self.batch_step
                elif sm_index < len(self.sm_clocks) - 1:
                    sm_index += step
            else:
                sm_index += step
        elif max_power>self.alpha*self.power_cap and max_power<=1.02*self.power_cap:
            pass
        else:
            if lantecy>self.belta*self.slo:
                batch_index -= step
            elif lantecy<=self.belta*self.slo:
                sm_index -= step
        sm_index = min(max(sm_index,0),len(self.sm_clocks)-1)
        batch_index = min(max(batch_index,0),len(self.batch_size_list)-1)

        return self.sm_clocks[sm_index], self.batch_size_list[batch_index]

class BatchDVFS:
    def __init__(self, power_cap, alpha, sm_clocks, batch_size_list, log_txt):
        self.power_cap = power_cap
        self.alpha = alpha
        self.sm_clocks = sm_clocks
        self.batch_size_list = batch_size_list
        self.log_txt = log_txt
        self.coarse_sm_step = 4
        self.fine_sm_step = 1
        self.min_bs_idx = 0
        self.max_bs_idx = len(batch_size_list) - 1
        self.current_bs_idx = 0
        self.current_sm_idx = len(sm_clocks) // 2 
        
    def scheduler(self, max_power, frequency, batch_size):
        try:
            self.current_bs_idx = self.batch_size_list.index(batch_size)
        except ValueError:
            self.current_bs_idx = 0

        try:
            self.current_sm_idx = self.sm_clocks.index(frequency)
        except ValueError:
            self.current_sm_idx = len(self.sm_clocks) // 2
        is_converged = (self.max_bs_idx - self.min_bs_idx) <= 1

        # --- 核心调度逻辑 (Algorithm 1) ---
        near_cap = abs(max_power - self.power_cap) <= 15.0
        sm_step = self.fine_sm_step if near_cap else self.coarse_sm_step

        # Case 1: 稳定区 [alpha * Cap, Cap) -> 保持现状
        if self.power_cap * self.alpha <= max_power < self.power_cap:
            pass 

        elif max_power < self.power_cap * self.alpha:
            if is_converged:
                if self.current_sm_idx < len(self.sm_clocks) - 1:
                    self.current_sm_idx += sm_step
                elif self.current_bs_idx < len(self.batch_size_list) - 1:
                    self.max_bs_idx = len(self.batch_size_list) - 1
            
            elif self.current_bs_idx < len(self.batch_size_list) - 1:
                self.min_bs_idx = self.current_bs_idx
                next_bs_idx = math.ceil((self.min_bs_idx + self.max_bs_idx) / 2)
                if next_bs_idx == self.current_bs_idx: 
                    next_bs_idx += 1
                self.current_bs_idx = next_bs_idx
            
            else:
                if self.current_sm_idx < len(self.sm_clocks) - 1:
                    self.current_sm_idx += sm_step
        
        else: 
            if is_converged:
                self.current_bs_idx = self.min_bs_idx
                
                if self.current_sm_idx > 0:
                    self.current_sm_idx -= sm_step
            
            elif self.current_bs_idx > 0:
                self.max_bs_idx = self.current_bs_idx
                next_bs_idx = math.floor((self.min_bs_idx + self.max_bs_idx) / 2)
                if next_bs_idx == self.current_bs_idx: 
                    next_bs_idx -= 1
                self.current_bs_idx = next_bs_idx
            else:
                if self.current_sm_idx > 0:
                    self.current_sm_idx -= sm_step
        self.current_bs_idx = max(0, min(self.current_bs_idx, len(self.batch_size_list) - 1))
        self.current_sm_idx = max(0, min(self.current_sm_idx, len(self.sm_clocks) - 1))

        return self.sm_clocks[self.current_sm_idx], self.batch_size_list[self.current_bs_idx]

class ThermoPowerScheduler:
    """
    Thermodynamic Model-Predictive Control wrapper for SM frequency updates.
    """
    def __init__(self, controller, log_txt):
        self.controller = controller
        self.log_txt = log_txt

    def scheduler(self, current_power, current_sm, p_list, b_list, dt=1.0, u_sm=None, u_mem=None):
        state = self.controller.step(
            f_curr=current_sm,
            p_meas=current_power,
            dt=dt,
            u_sm=u_sm,
            u_mem=u_mem,
            p_list=p_list,
            b_list=b_list,
            n_tasks=len(p_list),
        )
        return state.f_next
        
