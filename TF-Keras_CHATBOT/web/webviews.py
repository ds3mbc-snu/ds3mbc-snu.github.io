from django.shortcuts import render
from django.http import HttpResponse


from django.core.wsgi import get_wsgi_application

from django.urls import path





"""
Web-based demo
"""
import glob
import numpy as np

from django0 import MemN2N
from babi_util import parse_babi_task

memn2n = None
test_story, test_questions, test_qstory = None, None, None

		
		
		
#@app.route('/get/story', methods=['GET'])
def get_story():
    question_idx      = np.random.randint(test_questions.shape[1])
    story_idx         = test_questions[0, question_idx]
    last_sentence_idx = test_questions[1, question_idx]

    story_txt, question_txt, correct_answer = memn2n.get_story_texts(test_story, test_questions, test_qstory,
                                                                     question_idx, story_idx, last_sentence_idx)
    # Format text
    story_txt = "\n".join(story_txt)
    question_txt += "?"

    return flask.jsonify({
        "question_idx": question_idx,
        "story": story_txt,
        "question": question_txt,
        "correct_answer": correct_answer
    })

	
	
	

#@app.route('/get/answer', methods=['GET'])
def get_answer():
    question_idx  = int(flask.request.args.get('question_idx'))
    user_question = flask.request.args.get('user_question', '')

    story_idx         = test_questions[0, question_idx]
    last_sentence_idx = test_questions[1, question_idx]

    pred_answer_idx, pred_prob, memory_probs = memn2n.predict_answer(test_story, test_questions, test_qstory,
                                                                     question_idx, story_idx, last_sentence_idx,
                                                                     user_question)
    pred_answer = memn2n.reversed_dict[pred_answer_idx]

    return flask.jsonify({
        "pred_answer" : pred_answer,
        "pred_prob" : pred_prob,
        "memory_probs": memory_probs.T.tolist()
    })



def index(request):
    #run_web_demo(data_dir, model_file):
	#flaskapp.init(data_dir, model_file)
    #flaskapp.run()

    """ Initialize web app """
    global memn2n, test_story, test_questions, test_qstory
    data_dir=""
    model_file=""
	
    # Try to load model
    memn2n = MemN2N(data_dir, model_file)
    memn2n.load_model()

    # Read test data
    print("Reading test data from %s ..." % memn2n.data_dir)
    test_data_path = glob.glob('%s/qa*_*_test.txt' % memn2n.data_dir)
    test_story, test_questions, test_qstory = \
        parse_babi_task(test_data_path, memn2n.general_config.dictionary, False)



    return render(request, 'web/webindex.html')
	