#!/usr/bin/env python
# encoding: utf-8

__author__ = "yuzt"
__email__ = "zhenting.yu@gmail.com"


from heapq import heappush, heappop, heappushpop
from optparse import OptionParser
import sys
import string
import cPickle as pickle    #cPickle is an implementation of pickle in c
from evaluate import evaluate
from util import strdecode
import re
import codecs

reload(sys)
sys.setdefaultencoding('utf-8')

__boc__ = "__boc__"
__eoc__ = "__eoc__"
__bot__ = "__bot__"
__eot__ = "__eot__"
__bow__ = "__bow__"

digits = set(map(unicode, string.digits))
latins = set(map(unicode, string.ascii_letters))
en_punctations = {u"　", u"！", u"＂", u"＃", u"＄", u"％", u"＆", u"＇", u"（", u"）",
                    u"＊", u"＋", u"，", u"－", u"．", u"／", u"：", u"；", u"＜", u"＝",
                    u"＞", u"？", u"＠", u"［", u"＼", u"］", u"＾", u"＿", u"｀", u"｛",
                    u"｜", u"｝", u"～"}
ch_punctations = {u"。", u"、", u"“",  u"”",  u"﹃", u"﹄", u"‘",  u"’",  u"﹁", u"﹂",
                  u"…", u"【", u"】", u"《", u"》", u"〈", u"〉", u"·"}

re_eng = re.compile(ur"([A-Za-z0-9\.\+\-]*[A-za-z0-9])")
#re_url = re.compile(ur"((https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]*[-A-Za-z0-9+&@#/%=~_|])")
re_templates = [re_eng]


class State(object):
    '''
    The state object
    '''
    def __init__(self, score, index, state, action):
        self.score = score
        self.index = index
        self.link = state
        self.action = action

        if action == 'a':
            self.prev = state.prev
            self.curr = state.curr
        elif action == 's':
            self.prev = state.curr
            self.curr = self
        else:
            self.prev = None
            self.curr = self

    def empty(self):
        if self.index == -1:
            return True

    def __str__(self):
        ret = "ref: " + str(id(self))
        ret += " , index: " + str(self.index)
        ret += " , score:" + str(self.score)
        ret += " , prev:" + str(id(self.prev))
        ret += " , curr: " + str(id(self.curr))
        ret += " , action:" + self.action
        return str(ret)

    def __repr__(self):
        return self.__str__()

    def __lt__(self, other):
        return self.score < other.score

    def __gt__(self, other):
        return self.score > other.score

    def __eq__(self, other):
        return id(self) == id(other)


class Prune(object):
    NoAction = 0
    Append = 1
    Seperate = 2


class Rule(object):
    def __init__(self, raw_sentence):
        length = len(raw_sentence)
        rules = [Prune.NoAction for i in xrange(length+1)]

        for template in re_templates:
            for m in template.finditer(raw_sentence):
                start = m.start()
                end = m.end()
                rules[start] = Prune.Seperate
                for i in range(start+1, end):
                    rules[i] = Prune.Append

        #process space
        for i in range(length):
            if raw_sentence[i] == u" ":
                rules[i] = Prune.Seperate
        self.rules = rules

    def can_append(self,index):
        if self.rules[index] != Prune.Seperate:
            return True
        else:
            return False

    def can_seperate(self, index):
        if self.rules[index] != Prune.Append:
            return True
        else:
            return False

#rule = Rule(u"快上百度wwww.baidu.com!")
#print rule.rules
rule = Rule(u"今天是3月8号")
print rule.rules
#print len(rule.rules)


def chartype(ch):
    if ch in digits:
        return 'digit'
    elif ch in latins:
        return 'latin'
    elif ch in en_punctations or ch in ch_punctations:
        return 'punctation'
    else:
        return 'other'


def get_gold_actions(words):
    ret = ['s']
    for word in words:
        chars = word
        for idx, ch in enumerate(chars):
            if idx == len(chars) - 1:
                ret.append('s')     #'s' stands for seperate
            else:
                ret.append('a')     #'a' stands for append
    return ret

#print get_gold_actions([u'浦东',u'开发', u'与', u'法制',u'建设', u'同'])

def actions_to_words(actions, raw_sentence):
    words = []
    chars = [ch for ch in raw_sentence.decode('utf-8')]
    word = ''
    for idx, action in enumerate(actions):
        if idx == 0:
            word = chars[0]
            continue
        if action == 'a':
            word += chars[idx]
        elif action == 's':
            words.append(word)
            if idx < len(chars):
                word = chars[idx]
    return words

#print ' '.join(actions_to_words(['s','a','s','a','s','s','a','s'],'浦东开发与法制'))
#print ' '.join(actions_to_words(['s','a','s','a','s','s','s'],'浦东开发与法'))


def number_of_characters(words):
    '''
    Get number of characters in a list of words
    '''
    ret = sum(map(len, words))
    return ret

#print number_of_characters(['浦东', '开发', '与', '法制', '建设', '同步'])

def convert_words_to_characters(words):
    chars = []
    for word in words:
        chars.extend([ch for ch in word])
#    for ch in chars:
#        print ch
    return chars

#print convert_words_to_characters(['浦东', '开发', '与', '法制', '建设', '同步'])


def convert_raw_sentence_to_characters(raw_sentence):
    chars = []
    for ch in raw_sentence:
        chars.append(ch)
    return chars


def kmax_heappush(array, state, k):
    '''
    Inset the state into a k-max array
    '''
    if len(array) < k:
        heappush(array, state)
        return True
    elif len(array) == k:
        if array[0].score < state.score:
            heappushpop(array, state)
            return True
        return False
    return False


def extract_append_features(chars, charts, i, state):
    '''
    Extract features for JOIN action

    Paramters
    ---------
    chars : list(str)
        The list of characters
    charts : list(str)
        The list of character types
    i : int
        The index
    state : State
        The state

    Returns
    -------
    ret : list(str)
        The list of feature strings
    '''
    L = len(chars)

    prev_ch = chars[i-1] if i-1 >= 0 else __boc__
    curr_ch = chars[i]
    next_ch = chars[i+1] if i+1 < L else __eoc__
    prev_cht = charts[i - 1] if i - 1 >= 0 else __bot__
    curr_cht = charts[i]
    next_cht = charts[i + 1] if i + 1 < L else __eot__

    ret = ["1=c[-1]=%s" % prev_ch,
            "2=c[0]=%s" % curr_ch,
            "3=c[1]=%s" % next_ch,
            "4=ct[-1]=%s" % prev_cht,
            "5=ct[0]=%s" % curr_cht,
            "6=ct[1]=%s" % next_cht,
            "7=c[-1]c[0]=%s%s" % (prev_ch, curr_ch),
            "8=c[0]c[1]=%s%s" % (curr_ch, next_ch),
            "9=ct[-1]ct[0]=%s%s" % (prev_cht, curr_cht),
            "10=ct[0]ct[1]=%s%s" % (curr_cht, next_cht),
            "11=c[-1]c[0]c[1]=%s%s%s" % (prev_ch, curr_ch, next_ch),
            "12=ct[-1]ct[0]ct[1]=%s%s%s" % (prev_cht, curr_cht, next_cht)]
    ret = map(lambda x: x+u',action=a', ret)
    return ret


def extract_separate_features(chars, charts, i, state):
    '''
    Extract features for the SEPARATE actions

    Parameters
    ----------
    chars : list(str)
        The list of characters
    charts : list(str)
        The list of character types
    i : int
        The index
    state : State
        The source state

    Returns
    -------
    ret : list(str)
        The list of feature string
    '''
    L = len(chars)

    prev_ch = chars[i - 1] if i - 1 >= 0 else __boc__
    curr_ch = chars[i] if i < L else __eoc__
    next_ch = chars[i + 1] if i + 1 < L else __eoc__
    prev_cht = charts[i - 1] if i - 1 >= 0 else __bot__
    curr_cht = charts[i] if i< L else __eot__
    next_cht = charts[i + 1] if i + 1 < L else __eot__

    curr_w = "".join(chars[state.curr.index: i])
    curr_w_len = i  - state.curr.index
    if not state.prev.empty():
        prev_w = "".join(chars[state.prev.curr.index: state.curr.index])
        prev_w_len = state.curr.index - state.prev.index

    ret = ["1=c[-1]=%s" % prev_ch,
            "2=c[0]=%s" % curr_ch,
            "3=c[1]=%s" % next_ch,
            "4=ct[-1]=%s" % prev_cht,
            "5=ct[0]=%s" % curr_cht,
            "6=ct[1]=%s" % next_cht,
            "7=c[-1]c[0]=%s%s" % (prev_ch, curr_ch),
            "8=c[0]c[1]=%s%s" % (curr_ch, next_ch),
            "9=ct[-1]ct[0]=%s%s" % (prev_cht, curr_cht),
            "10=ct[0]ct[1]=%s%s" % (curr_cht, next_cht),
            "11=c[-1]c[0]c[1]=%s%s%s" % (prev_ch, curr_ch, next_ch),
            "12=ct[-1]ct[0]ct[1]=%s%s%s" % (prev_cht, curr_cht, next_cht)]
    ret = map(lambda x: x+u',action=s', ret)

    ret.append("13=w[0]=%s" % curr_w)
    if len(curr_w) == 0:
        pass
    if curr_w_len == 1:
        ret.append("14=single-char")
    else:
        ret.append("15=first=%s=last=%s" % (chars[state.curr.index], chars[i-1]))
        ret.append("16=first=%s=len[0]=%d" % (chars[state.curr.index], curr_w_len))
        ret.append("17=last=%s=len[0]=%d" % (chars[i-1], curr_w_len))

    if not state.prev.empty():
        ret.append("18=word[-1]=%s-word[0]=%s" % (prev_w, curr_w))
        ret.append("19=prevch=%s-word[0]=%s" % (chars[state.curr.index-1], curr_w))
        ret.append("20=word[-1]=%s-prevch=%s" % (prev_w, chars[state.curr.index-1]))
        ret.append("21=word[-1]=%s-len[0]=%d" % (prev_w, curr_w_len))
        ret.append("22=word[0]=%s-len[-1]=%d" % (curr_w, prev_w_len))
    if i < L-1:
        ret.append("23=c[-1]c[0]=%s%s" % (chars[i-1], chars[i]))
        ret.append("24=w[-1]=%s-c[0]=%s" % (curr_w, chars[1]))
        ret.append("25=first=%s-c[0]=%s" % (chars[state.curr.index], chars[i]))
    return ret


def extract_features(action, chars, charts, i, state):
    if action == 's':
        return extract_separate_features(chars, charts, i, state)
    else:
        return extract_append_features(chars, charts, i, state)


def transition_score(action, chars, charts, i, state, params, train=False):
    '''
    Compute transition score according to the action

    Parameters
    ----------
    state : State
        The state
    action : str
        The action str
    params : dict
        The model parameters
    chars : List
        Every ch in the raw_sentence
    charts : List
        Chartype of every ch in the raw_sentence

    Returns
    -------
    ret : float
        The transition score
    '''
    ret = 0.
    if state.empty():
        return ret

    L = len(chars)
    if action == 'a':
        for feature in extract_append_features(chars, charts, i, state):
            if feature in params:
                ret += params[feature][0] if train else params[feature][2]
    elif action == 's':
        for feature in extract_separate_features(chars, charts, i, state):
            if feature in params:
                ret += params[feature][0] if train else params[feature][2]
    return ret


def flush_parameters(params, now):
    '''
    At the end of each iteration, flash the parameters

    Parameters
    ----------
    param : dict
        The parameters
    now : int
        The current time
    '''
    for feature in params:
        w = params[feature]
        w[2] += (now - w[1]) * w[0]
        w[1] = now


#TODO lazy update
def update_parameters(params, features, now,  scale):
    '''
    Update the parameters
    Use lazy update strategy
    For each time there only a small proportion of features are coverd,
    which is more efficient than naive average method.
    '''
    for feature in features:
        if feature not in params:
            params[feature] = [0 for i in range(3)]

        w = params[feature]
        elapsed = now - w[1]
        upd = scale
        cur_val = w[0]

        w[0] = cur_val + upd                #update weights withou average
        w[2] += elapsed * cur_val + upd     #average weights
        w[1] = now                          #record last update time

#        if feature not in params:
#            params[feature] = 0.
#        params[feature] += scale


def backtrace_and_get_state(state):
    '''
    backtrace and get the action sequence
    '''
    ret = []
    while state.index > -1:
        ret.append(state)
        state = state.link
    ret.reverse()
    return ret


def beam_search(train, raw_sentence, beam_size, params, reference_actions=None, now=None):
    '''
    Parameters
    ----------
    train: bool
        If run training process
    words: list(str)
        The words list
    beam_size: int
        The size of beam
    params: dict
        The parameters
    reference_action: list(str)
        The reference actions

    Return
    ------
    (int, )
    '''
    # iniallize the beam matrix
    chars = convert_raw_sentence_to_characters(raw_sentence)
    charts = [chartype(ch) for ch in chars]
    L = len(chars)
    beam = [[] for _ in xrange(L+2)]
    beam[0].append(State(score=0., index=-1, state=None, action=None))
    correct_state = beam[0][0]

    rule = Rule(raw_sentence)
    for i in xrange(L+1):
        for state in beam[i]:
            if 0 < i < L and rule.can_append(i):
                gain = transition_score('a', chars, charts, i, state, params, train)
                added = kmax_heappush(beam[i+1],
                        State(score = state.score + gain,
                            index = i,
                            state = state,
                            action = 'a'),
                        beam_size)

            if rule.can_seperate(i):
                gain = transition_score('s', chars, charts, i, state, params, train)
                added = kmax_heappush(beam[i+1],
                        State(score = state.score + gain,
                            index = i,
                            state = state,
                            action = 's'),
                        beam_size)

        if train:
            in_beam = False
            for state in beam[i+1]:
                if state.link == correct_state and state.action == reference_actions[i]:
                    in_beam = True
                    correct_state = state
                    correct_state_path = backtrace_and_get_state(correct_state)
                    correct_actions = [state.action for state in correct_state_path]
                    break

            if not in_beam:
                # early update
                best_predict_state = max(beam[i+1], key=lambda s: s.score)
                predict_state_path = backtrace_and_get_state(best_predict_state)

                beam[i+1].append(State(score = 0, index = i , state = correct_state, action = reference_actions[i]))
                correct_state_path = backtrace_and_get_state(beam[i+1][-1])
                predict_actions = [state.action for state in predict_state_path]
                correct_actions = [state.action for state in correct_state_path]

                assert(len(predict_state_path) == len(correct_state_path))

                update_start_position = -1
                for i in xrange(len(predict_state_path) - 1):
                    if predict_state_path[i] == correct_state_path[i] and predict_state_path[i+1].action != correct_state_path[i+1].action:
                        update_start_position = i
                        break

                #for i in xrange(update_start_position, len(predict_state_path) - 1):
                for i in xrange(len(predict_state_path)-1):
                    correct_features = extract_features(correct_state_path[i+1].action, chars, charts, i+1, correct_state_path[i])
                    predict_features = extract_features(predict_state_path[i+1].action, chars, charts, i+1, predict_state_path[i])
                    update_parameters(params, correct_features, now,  1.)
                    update_parameters(params, predict_features, now, -1.)
                return

    if train:
        best_predict_state = max(beam[i+1],key=lambda s:s.score)
        predict_state_path = backtrace_and_get_state(best_predict_state)
        correct_state_path = backtrace_and_get_state(correct_state)
        predict_actions = [state.action for state in predict_state_path]
        correct_actions = [state.action for state in correct_state_path]
        correct_words = actions_to_words(correct_actions, raw_sentence)
        predict_words = actions_to_words(predict_actions, raw_sentence)
#        print ' '.join(correct_words)
#        print ' '.join(predict_words)
#        words = actions_to_words(predict_actions, raw_sentence)

        assert(len(predict_state_path) == len(correct_state_path))
        for i in xrange(L):
            correct_features = extract_features(correct_state_path[i+1].action, chars, charts, i+1, correct_state_path[i] )
            predict_features = extract_features(predict_state_path[i+1].action, chars, charts, i+1, predict_state_path[i])

            update_parameters(params, correct_features, now,  1.)
            update_parameters(params, predict_features, now, -1.)
        return
    else:
        best_predict_state = max(beam[i+1],key=lambda s:s.score)
        predict_state_path = backtrace_and_get_state(best_predict_state)
        actions = [state.action for state in predict_state_path]
        words = actions_to_words(actions, raw_sentence)
        return words


def learn(opts):
    if opts.train:
        train(opts)
    elif opts.test:
        segment(opts)


def train(opts):
    train_file = opts.train
    model_file = opts.model
    beam_size = opts.beam_size
    assert (train_file != "")
    assert (model_file != "")
    assert (beam_size >= 1)
#    if not os.path.exits(model_file):
#        params = {}
#    else:
#        params = pickle.load(open(model_file))
    params = {}
    data = []
    with codecs.open(train_file, encoding='utf-8') as infile:
        for line in infile:
            words = line.strip().split()
            data.append(words)
    num_instances = len(data)
    if opts.dev:
        dev_data = []
        dev_file = opts.dev
        with codecs.open(dev_file, encoding='utf-8') as infile:
            for line in infile:
                words = line.strip().split()
                dev_data.append(words)


    for nr_iter in xrange(opts.iteration):
        for idx, sentence in enumerate(data):
            now = nr_iter * num_instances + 1
            gold_actions = get_gold_actions(sentence)
            raw_sentence = ''.join(sentence)
            beam_search(True, raw_sentence, beam_size, params, gold_actions, now)

        temp_model_file = "{0}.{1}".format(model_file, nr_iter+1)
        assert(len(params)>0)
        flush_parameters(params, now)
        pickle.dump(params, open(temp_model_file, 'w'))
        if opts.dev:
            temp_file = "temp/temp%s" % nr_iter
            with open(temp_file, 'w') as outfile:
                for sentence in dev_data:
                    raw_sentence = ''.join(sentence)
                    words = beam_search(False, raw_sentence, beam_size, params)
                    line = ' '.join(words)
                    line += '\n'
#                    print raw_sentence
#                    print line.strip()
                    outfile.write(line)
            p, r, f = evaluate(temp_file, dev_file)
            print "Precision:{0}, Recall:{1}, Fscore:{2}".format(p, r, f)


def segment(opts):
    test_file = opts.test
    model_file = opts.model
    output_file = opts.output
    beam_size = opts.beam_size
    assert (test_file != "")
    assert (model_file != "")
    assert (output_file != "")
    assert (beam_size >= 1)
    params = pickle.load(open(model_file))
    output= []
    #decode
    with codecs.open(test_file, encoding='utf-8') as infile:
        for line in infile:
            line = strdecode(line)
            raw_sentence = line.strip()
            words = beam_search(False, raw_sentence, beam_size, params)
            output.append(words)
    #save segmented sentences to output_file
    with open(output_file) as outfile:
        for words in output:
            line = ' '.join(words)
            line += '\n'
            outfile.write(line)


if __name__=="__main__":
    usage = "A Python implementation for Zhang and Clark (2007)"
    optparser = OptionParser(usage)
    optparser.add_option("-t", "--train", dest="train", help="use to specify training data")
    optparser.add_option("-d", "--dev", dest="dev", help="use to specify development data")
    optparser.add_option("-e", "--test", dest="test", help="use to specify test data")
    optparser.add_option("-i", "--iteration", dest="iteration", type=int, help="use to specify the maximum number of iteration")
    optparser.add_option("-b", "--beam_size", dest="beam_size", type=int, help="use to specify the size of the beam")
    optparser.add_option("-m", "--model", dest="model", help="use to sepcify model file")
    optparser.add_option("-o", "--ouput", dest="output", help="use to save the output")
    opts, args = optparser.parse_args()

    print opts
    learn(opts)

