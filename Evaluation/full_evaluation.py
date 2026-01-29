# This code is from: https://github.com/anniedoris/design_qa/


import argparse
import os
from datetime import datetime
from metrics import eval_retrieval_qa, eval_compilation_qa, eval_definition_qa, eval_presence_qa, eval_dimensions_qa, eval_functional_performance_qa

def save_results(model, macro_avg_accuracy, all_accuracies, macro_avg_bleus, all_bleus, macro_avg_rogues, all_rogues, sim, question_type = "functional_performance"):
    
    print(f"Model: {model}")
    print(f"\nMacro avg: {macro_avg_accuracy}")
    print(f"\nAll accuracies: {all_accuracies}")
    print(f"\nMacro avg bleus: {macro_avg_bleus}")
    print(f"\nAll bleus: {all_bleus}")
    print(f"\nMacro avg rogues: {macro_avg_rogues}")
    print(f"\nAll rogues: {all_rogues}")

    # Save results to txt file
    with open(f"EvalResults_SAM/{question_type}_evaluation_{model}_test_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt", "w") as text_file:
        text_file.write(f"Model: {model}")
        text_file.write(f"\nMacro avg: {macro_avg_accuracy}")
        text_file.write(f"\nAll accuracies: {all_accuracies}")
        text_file.write(f"\nMacro avg bleus: {macro_avg_bleus}")
        text_file.write(f"\nAll bleus: {all_bleus}")
        text_file.write(f"\nMacro avg rogues: {macro_avg_rogues}")
        text_file.write(f"\nAll rogues: {all_rogues}")
        text_file.write(f"\nSimilarity Score: {sim}")

def test_functional(question_type = "functional_performance", path_to_csv = "results/fsae_test_rule_functional_performance_qa.csv"):
    print(f"test {question_type}")
    res = eval_functional_performance_qa(path_to_csv)
    if len(res) == 7:
        macro_avg_accuracy, all_accuracies, macro_avg_bleus, all_bleus, macro_avg_rogues, all_rogues, sim = res
    elif len(res) == 6:
        macro_avg_accuracy, all_accuracies, macro_avg_bleus, all_bleus, macro_avg_rogues, all_rogues = res
        sim = "N/A"
    else:
        raise ValueError(f"Unexpected return length from eval_functional_performance_qa: {len(res)}")

    save_results(question_type, macro_avg_accuracy, all_accuracies, macro_avg_bleus, all_bleus, macro_avg_rogues, all_rogues, sim)


def test_dimension(path_to_csv = "results/fsae_test_rule_functional_performance_qa.csv", detailed_context = False):

    macro_avg_accuracy, direct_dim_avg, scale_bar_avg, all_accuracies, macro_avg_bleus, all_bleus, \
            macro_avg_rogues, all_rogues = eval_dimensions_qa(path_to_csv)
    print(f"\nMacro avg: {macro_avg_accuracy}")
    print(f"\nDirect Dimension avg: {direct_dim_avg}")
    print(f"\nScale Bar avg: {scale_bar_avg}")
    print(f"\nAll accuracies: {all_accuracies}")
    print(f"\nMacro avg bleus: {macro_avg_bleus}")
    print(f"\nAll bleus: {all_bleus}")
    print(f"\nMacro avg rogues: {macro_avg_rogues}")
    print(f"\nAll rogues: {all_rogues}")

    # Save results to txt file
    with open(f"EvalResults_SAM/dimension_evaluation_{'detailed_context' if detailed_context else 'context'}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt", "w") as text_file:
        text_file.write(f"\nMacro avg: {macro_avg_accuracy}")
        text_file.write(f"\nDirect Dimension avg: {direct_dim_avg}")
        text_file.write(f"\nScale Bar avg: {scale_bar_avg}")
        text_file.write(f"\nAll accuracies: {all_accuracies}")
        text_file.write(f"\nMacro avg bleus: {macro_avg_bleus}")
        text_file.write(f"\nAll bleus: {all_bleus}")
        text_file.write(f"\nMacro avg rogues: {macro_avg_rogues}")
        text_file.write(f"\nAll rogues: {all_rogues}")

def test_definition(path_to_csv = "results/fsae_test_rule_functional_performance_qa.csv"):
    macro_avg_accuracy, definitions_qs_definition_avg, multi_qs_definition_avg, single_qs_definition_avg, all_answers_definition = eval_definition_qa(path_to_csv)
    print(f"\nMacro avg: {macro_avg_accuracy}")
    print(f"\nDefinitions: {definitions_qs_definition_avg}")
    print(f"\nMulti avg: {multi_qs_definition_avg}")
    print(f"\nSingle avg: {single_qs_definition_avg}")
    print(f"\nAll answers: {all_answers_definition}")

    # Save results to txt file
    with open(f"EvalResults_SAM/definition_evaluation_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt", "w") as text_file:
        text_file.write(f"\nMacro avg: {macro_avg_accuracy}")
        text_file.write(f"\nDefinitions: {definitions_qs_definition_avg}")
        text_file.write(f"\nMulti avg: {multi_qs_definition_avg}")
        text_file.write(f"\nSingle avg: {single_qs_definition_avg}")
        text_file.write(f"\nAll answers: {all_answers_definition}")

def test_presence(path_to_csv = "results/fsae_test_rule_functional_performance_qa.csv"):
    macro_avg_accuracy, definitions_qs_presence_avg, multi_qs_presence_avg, single_qs_presence_avg, all_answers_presence = eval_presence_qa(path_to_csv)
    print(f"\nMacro avg: {macro_avg_accuracy}")
    print(f"\nDefinitions: {definitions_qs_presence_avg}")
    print(f"\nMulti avg: {multi_qs_presence_avg}")
    print(f"\nSingle avg: {single_qs_presence_avg}")
    print(f"\nAll answers: {all_answers_presence}")

    # Save results to txt file
    with open(f"EvalResults_SAM/presence_evaluation_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt", "w") as text_file:
        text_file.write(f"\nMacro avg: {macro_avg_accuracy}")
        text_file.write(f"\nDefinitions: {definitions_qs_presence_avg}")
        text_file.write(f"\nMulti avg: {multi_qs_presence_avg}")
        text_file.write(f"\nSingle avg: {single_qs_presence_avg}")
        text_file.write(f"\nAll answers: {all_answers_presence}")

def test_retrieval(path_to_csv = "results/fsae_test_rule_functional_performance_qa.csv"):
    macro_avg_accuracy, all_answers_retrieval = eval_retrieval_qa(path_to_csv)
    print(f"\nMacro avg: {macro_avg_accuracy}")
    print(f"\nAll answers: {all_answers_retrieval}")
    
    # Save results to txt file
    with open(f"EvalResults_SAM/retrieval_evaluation_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt", "w") as text_file:
        text_file.write(f"\nMacro avg: {macro_avg_accuracy}")
        text_file.write(f"\nAll answers: {all_answers_retrieval}")

def test_compilation(path_to_csv = "results/fsae_test_rule_functional_performance_qa.csv"):
    macro_avg_accuracy, all_answers_compilation = eval_compilation_qa(path_to_csv)
    print(f"\nMacro avg: {macro_avg_accuracy}")
    print(f"\nAll answers: {all_answers_compilation}")
    
    # Save results to txt file
    with open(f"EvalResults_SAM/compilation_evaluation_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt", "w") as text_file:
        text_file.write(f"\nMacro avg: {macro_avg_accuracy}")
        text_file.write(f"\nAll answers: {all_answers_compilation}")

def main():
    parser = argparse.ArgumentParser(description="Optional paths for CAD evaluation inputs")

    parser.add_argument("--path_to_retrieval", type=str, default="results/rule_retrieval_qa_with_predictions.csv",
                        help="Path to csv containing retrieval data (optional)")
    parser.add_argument("--path_to_compilation", type=str, default="results/rule_compilation_qa_with_predictions.csv",
                        help="Path to csv containing compilation data (optional)")
    parser.add_argument("--path_to_dimension", type=str, default="results/rule_dimension_qa_context_with_predictions.csv",
                        help="Path to csv containing dimension data (optional)")
    parser.add_argument("--path_to_dimension_detailed_context", type=str, default="results/rule_dimension_qa_detailed_context_with_predictions.csv",
                        help="Path to csv containing dimension data (optional)")
    parser.add_argument("--path_to_functional_performance", type=str, default="results/rule_functional_performance_qa_with_predictions.csv",
                        help="Path to csv containing functional performance data (optional)")
    parser.add_argument("--path_to_definition", type=str, default="results/rule_definition_qa_with_predictions.csv",
                        help="Path to csv containing definition data (optional)")
    parser.add_argument("--path_to_presence", type=str, default="results/rule_presence_qa_with_predictions.csv",
                        help="Path to csv containing presence data (optional)")
    parser.add_argument("--save_path", type=str, default="results.txt",
                        help="Path to .txt file to save the evaluation results (default: results.txt)")

    args = parser.parse_args()

    print("Arguments received:")
    print(f"  path_to_retrieval: {args.path_to_retrieval}")
    print(f"  path_to_compilation: {args.path_to_compilation}")
    print(f"  path_to_dimension: {args.path_to_dimension}")
    print(f"  path_to_dimension_detailed_context: {args.path_to_dimension_detailed_context}")
    print(f"  path_to_functional_performance: {args.path_to_functional_performance}")
    print(f"  path_to_definition: {args.path_to_definition}")
    print(f"  path_to_presence: {args.path_to_presence}")
    
    all_subsets = []
    if args.path_to_retrieval:
        macro_avg_retrieval, all_answers_retrieval = eval_retrieval_qa(args.path_to_retrieval)
        all_subsets.append(macro_avg_retrieval)
        
    if args.path_to_compilation:
        macro_avg_compilation, all_answers_compilation = eval_compilation_qa(args.path_to_compilation)
        all_subsets.append(macro_avg_compilation)
        
    if args.path_to_definition:
        macro_avg_definition, definitions_qs_definition_avg, multi_qs_definition_avg, single_qs_definition_avg, all_answers_definition = eval_definition_qa(args.path_to_definition)
        all_subsets.append(macro_avg_definition)
        
    if args.path_to_presence:
        macro_avg_presence, definitions_qs_presence_avg, multi_qs_presence_avg, single_qs_presence_avg, all_answers_presence = eval_presence_qa(args.path_to_presence)
        all_subsets.append(macro_avg_presence)
        
    if args.path_to_dimension:
        macro_avg_accuracy_dimension, direct_dim_avg, scale_bar_avg, all_accuracies_dimension, macro_avg_bleus_dimension, all_bleus_dimension, \
                macro_avg_rogues_dimension, all_rogues_dimension = eval_dimensions_qa(args.path_to_dimension)
        all_subsets.append(macro_avg_accuracy_dimension)

    if args.path_to_dimension_detailed_context:
        macro_avg_accuracy_dimension_detailed_context, direct_dim_avg_detailed_context, scale_bar_avg_detailed_context, all_accuracies_dimension_detailed_context, macro_avg_bleus_dimension_detailed_context, all_bleus_dimension_detailed_context, \
                macro_avg_rogues_dimension_detailed_context, all_rogues_dimension_detailed_context = eval_dimensions_qa(args.path_to_dimension_detailed_context)
        all_subsets.append(macro_avg_accuracy_dimension_detailed_context)

    if args.path_to_functional_performance:
        macro_avg_accuracy_functional, all_accuracies_functional, macro_avg_bleus_functional, all_bleus_functional, macro_avg_rogues_functional, all_rogues_functional,sim = eval_functional_performance_qa(args.path_to_functional_performance)
        all_subsets.append(macro_avg_accuracy_functional)

    # Write all the results to a file
    with open(args.save_path, 'w') as text_file:
        text_file.write("DESIGNQA EVALUATION RESULTS:\n")
        text_file.write("-*-" * 20 + "\n")
        text_file.write("-*-" * 20 + "\n")
        text_file.write(f"OVERALL SCORE: {sum(all_subsets) / 6}\n")
        text_file.write("-*-" * 20 + "\n")
        text_file.write("-*-" * 20 + "\n")
        text_file.write(f"Retrieval Score (Avg F1 BoW): {macro_avg_retrieval if args.path_to_retrieval else 'N/A'}\n")
        text_file.write(f"Compilation Score (Avg F1 Rules): {macro_avg_compilation if args.path_to_compilation else 'N/A'}\n")
        text_file.write(f"Definition Score (Avg F1 BoC): {macro_avg_definition if args.path_to_definition else 'N/A'}\n")
        text_file.write(f"Presence Score (Avg Accuracy): {macro_avg_presence if args.path_to_presence else 'N/A'}\n")
        text_file.write(f"Dimension Score (Average Accuracy): {macro_avg_accuracy_dimension if args.path_to_dimension else 'N/A'}\n")
        text_file.write(f"Dimension Score (Average Accuracy): {macro_avg_accuracy_dimension_detailed_context if args.path_to_dimension_detailed_context else 'N/A'}\n")
        text_file.write(f"Functional Performance Score (Average Accuracy): {macro_avg_accuracy_functional if args.path_to_functional_performance else 'N/A'}\n")
        text_file.write("-*-" * 20 + "\n")
        text_file.write("\n\n\n")
        text_file.write("Below scores by subset are provided for diagnostic purposes:\n")
        text_file.write("---" * 20 + "\n")
        text_file.write("RETRIEVAL\n")
        text_file.write("---" * 20 + "\n")
        if args.path_to_retrieval:
            text_file.write(f"All F1 BoWs:\n{all_answers_retrieval}\n")
        else:
            text_file.write("No retrieval data provided.\n")
        
        text_file.write("---" * 20 + "\n")
        text_file.write("COMPILATION\n")
        text_file.write("---" * 20 + "\n")
        if args.path_to_compilation:
            text_file.write(f"All F1 Rules:\n{all_answers_compilation}\n")
        else:
            text_file.write("No compilation data provided.\n")
        
        text_file.write("---" * 20 + "\n")
        text_file.write("DEFINITION\n")
        text_file.write("---" * 20 + "\n")
        if args.path_to_definition:
            text_file.write(f"Avg F1 BoC on definition-components:\n{definitions_qs_definition_avg}\n")
            text_file.write(f"Avg F1 BoC on multimention-components:\n{multi_qs_definition_avg}\n")
            text_file.write(f"Avg F1 BoC on no-mention-components:\n{single_qs_definition_avg}\n")
            text_file.write(f"All F1 BoC:\n{all_answers_definition}\n")
        else:
            text_file.write("No definition data provided.\n")
            
        text_file.write("---" * 20 + "\n")
        text_file.write("PRESENCE\n")
        text_file.write("---" * 20 + "\n")
        if args.path_to_presence:
            text_file.write(f"Avg accuracy on definition-components:\n{definitions_qs_presence_avg}\n")
            text_file.write(f"Avg accuracy on multimention-components:\n{multi_qs_presence_avg}\n")
            text_file.write(f"Avg accuracy on no-mention-components:\n{single_qs_presence_avg}\n")
            text_file.write(f"All accuracies:\n{all_answers_presence}\n")
        else:
            text_file.write("No presence data provided.\n")

        text_file.write("---" * 20 + "\n")
        text_file.write("DIMENSION\n")
        text_file.write("---" * 20 + "\n")
        if args.path_to_dimension:
            text_file.write(f"Avg accuracy directly-dimensioned:\n{direct_dim_avg}\n")
            text_file.write(f"Avg accuracy scale-bar-dimensioned:\n{scale_bar_avg}\n")
            text_file.write(f"All accuracies:\n{all_accuracies_dimension}\n")
            text_file.write(f"Avg BLEU score:\n{macro_avg_bleus_dimension}\n")
            text_file.write(f"All BLEU scores:\n{all_bleus_dimension}\n")
            text_file.write(f"Avg ROUGE score:\n{macro_avg_rogues_dimension}\n")
            text_file.write(f"All ROUGE scores:\n{all_rogues_dimension}\n")
        else:
            text_file.write("No dimension data provided.\n")

        text_file.write("---" * 20 + "\n")
        text_file.write("DIMENSION\n")
        text_file.write("---" * 20 + "\n")
        if args.path_to_dimension_detailed_context:
            text_file.write(f"Avg accuracy directly-dimensioned:\n{direct_dim_avg_detailed_context}\n")
            text_file.write(f"Avg accuracy scale-bar-dimensioned:\n{scale_bar_avg_detailed_context}\n")
            text_file.write(f"All accuracies:\n{all_accuracies_dimension_detailed_context}\n")
            text_file.write(f"Avg BLEU score:\n{macro_avg_bleus_dimension_detailed_context}\n")
            text_file.write(f"All BLEU scores:\n{all_bleus_dimension_detailed_context}\n")
            text_file.write(f"Avg ROUGE score:\n{macro_avg_rogues_dimension_detailed_context}\n")
            text_file.write(f"All ROUGE scores:\n{all_rogues_dimension_detailed_context}\n")
        else:
            text_file.write("No dimension data provided.\n")
            
        text_file.write("---" * 20 + "\n")
        text_file.write("FUNCTIONAL PERFORMANCE\n")
        text_file.write("---" * 20 + "\n")
        if args.path_to_functional_performance:
            text_file.write(f"All accuraciess:\n{all_accuracies_functional}\n")
            text_file.write(f"Avg BLEU score:\n{macro_avg_bleus_functional}\n")
            text_file.write(f"All BLEU scores:\n{all_bleus_functional}\n")
            text_file.write(f"Avg ROUGE score:\n{macro_avg_rogues_functional}\n")
            text_file.write(f"All ROUGE scores:\n{all_rogues_functional}\n")
            text_file.write(f"\nSimilarity Score: {sim}")
        else:
            text_file.write("No functional performance data provided.\n")

if __name__ == "__main__":
    #main()
    # filedirresult="Final/high_reasoning"
    # name="_high_reasoning_gpt5mini"
    filedirresult="results"
    name=""
    test_compilation(path_to_csv = f"{filedirresult}/rule_compilation_qa_with_predictions{name}.csv")
    test_presence(path_to_csv = f"{filedirresult}/rule_presence_qa_with_predictions{name}.csv")
    test_definition(path_to_csv = f"{filedirresult}/rule_definition_qa_with_predictions{name}.csv")
    #test_dimension(path_to_csv = f"{filedirresult}/rule_dimension_qa_context_with_predictions_New.csv", detailed_context = False)
    test_dimension(path_to_csv = f"{filedirresult}/rule_dimension_qa_detailed_context_with_predictions{name}.csv", detailed_context = True)
    test_functional(path_to_csv = f"{filedirresult}/rule_functional_performance_qa_with_predictions{name}.csv")
    test_retrieval(path_to_csv = f"{filedirresult}/rule_retrieval_qa_with_predictions{name}.csv")
    





