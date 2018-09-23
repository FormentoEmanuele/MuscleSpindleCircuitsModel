: Mechanism for a model of the afferent fiber.
: Emanuele Formento.

NEURON {
	ARTIFICIAL_CELL AfferentFiber
	RANGE EES
	RANGE fireFlag, tLastSpike : To plot membrane state as in other artificial cells
}

PARAMETER {
	EES = 0
}

ASSIGNED {
	fireFlag
	tLastSpike
}

INITIAL {
	fireFlag = 0
	tLastSpike = 0
}

FUNCTION M(){
	M = fireFlag
	:printf("event %g\n",fireFlag)
	if (t-tLastSpike>1){
		fireFlag = 0
	}
}

NET_RECEIVE (w){
	if (w==-3){
		EES=1
	}
	if (w!=-3 && flag==1){
		net_event(t)
		fireFlag = 1
		tLastSpike = t
	}
}
