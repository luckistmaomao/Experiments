#coding:utf-8

from zhang2007_impl import kmax_heappush, State, Rule

def test():
    array = []
    prev = State(0,0,None,'h')
    for i in range(10):
        state = State(i,1,prev,'s')
        kmax_heappush(array,state,4)
#        print ' '.join([str(state.score) for state in array])
    print 'end'
    for state in array:
        if state.prev == prev:
            print 'yes'

def test_rule():
    sent = u'今天是2014年8月10号'
    r = Rule(sent)
    print r.rules

if __name__ == "__main__":
    test_rule()
