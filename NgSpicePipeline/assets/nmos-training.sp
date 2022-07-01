nmos: cs amp

.include {model_path}

.option TEMP=27C

V6 net2 0 dc 822.6m ac 1
Vdd net1 0 dc 1.2
R0 net1 out 500
mn0 out net2 0 0 NMOS W=5u L=45n

.control

let w_start = {start1}
let w_stop = {stop1}
let delta_w = {change1}
let w_test = w_start
let r_start = {start2}
let r_stop = {stop2}
let delta_r = {change2}
let r_test = r_start

while w_test le w_stop
	alter @mn0[w] = w_test
	while r_test le r_stop
		alter R0 = r_test
		op
		let gm = @mn0[gm]
		let ro = 1/@mn0[gds]
		let a0 = gm*r_test
		let id = @mn0[id]
		let pw = id*1.2
		let cgd = @mn0[cgd]
		let cdb = abs(@mn0[cdb])
		let cp = (1+a0)*cgd+cdb
		let bw = 1/(2*pi*r_test*cp)
		wrdata {out}bw.csv bw
		wrdata {out}pw.csv pw
		wrdata {out}a0.csv a0
		set appendwrite
		let r_test = r_test + delta_r	
	end
	let w_test = w_test + delta_w
	let r_test = r_start	
end

.endc

.end