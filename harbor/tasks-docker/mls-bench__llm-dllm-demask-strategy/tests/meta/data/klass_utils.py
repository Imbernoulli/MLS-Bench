import json
import re
import ast
import regex
import torch
import multiprocessing

def process_file(filename):
    with open(filename, 'r') as file:
        return [json.loads(line) for line in file if line.strip()]
    

### GSM8K Dataset Parsing Functions ###

def parse_answer(string) -> str:
    """
    Parse and extract answer from a string

    Parameters:
        string (str): The input string to parse
    
    Return:
        str: A string containing the parsed answer, or empty if the string does not match the expected format
    """
    
    # Split the string based on the phrase "Final answer:"
    string_parts = string.split("answer is")
    # Set a flag to indicate if there is an answer
    answer_flag = True if len(string_parts) > 1 else False
    # Get the last part of the string
    string = string_parts[-1]

    # Check for LaTeX boxed answers first
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', string)
    if boxed_match:
        # Extract only the numeric value from inside the box
        box_content = boxed_match.group(1).strip()
        numeric_match = re.search(r'-?\d+\.?\d*', box_content)
        if numeric_match:
            return numeric_match.group(0).replace(",", "")
        return box_content.replace(",", "")  # Fallback to full content if no number found
    
    # Remove comas from the string
    string = string.replace(",", "")
    
    # Find all numeric values in the string using regular expression
    string = [s for s in re.findall(r'-?\d+\.?\d*', string)]

    # If there are no candidates in the list, set string to an empty string
    if len(string) == 0:
        string = ""
    else:
        # Choose the first or last element based on the answer flag
        if answer_flag:
            string = string[0]
        else:
            string = string[-1]
    
    # Remove trailing period if present
    string = string.rstrip('.')
    string = string.replace("\n", "")
    # Remove trailing zeros from decimals
    if re.match(r'^-?\d+\.\d+$', string):
        string = string.rstrip('0').rstrip('.')
    
    return string

def parse_ground_truth_answer(string) -> str:
    answer_pattern = re.compile(r"#### (\-?[0-9\.\,]+)")
    match = answer_pattern.search(string)

    # Extract match and clean up
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        # Remove trailing zeros from decimals
        if re.match(r'^-?\d+\.\d+$', match_str):
            match_str = match_str.rstrip('0').rstrip('.')
    ground_truth_answer = match_str

    return ground_truth_answer


### MATH Dataset Parsing Functions ###

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _fix_sqrt(string):
    _string = re.sub(r"\\sqrt(-?[0-9.a-zA-Z]+)", r"\\sqrt{\1}", string)
    _string = re.sub(r"\\sqrt\s+(\w+)$", r"\\sqrt{\1}", _string)
    return _string


def _fix_tan(string):
    _string = re.sub(r"\\tan(-?[0-9.a-zA-Z]+)", r"\\tan{\1}", string)
    _string = re.sub(r"\\tan\s+(\w+)$", r"\\tan{\1}", _string)
    return _string


def strip_string(string):
    string = str(string).strip()
    # linebreaks
    string = string.replace("\n", "")

    # right "."
    string = string.rstrip(".")

    # remove inverse spaces
    string = string.replace("\\!", "")

    if string.startswith("\\text{") and string.endswith("}"):
        string = string.split("{", 1)[1][:-1]

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("cfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove unit: miles, dollars if after is not none
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        # print("Warning: unit not removed: '{}' -> '{}'".format(string, _string))
        string = _string

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "").strip()
    string = string.replace("^\\circ", "").strip()

    string = regex.sub(r"\{(c|m)?m\}(\^(2|3))?", "", string).strip()
    string = regex.sub(r"p\.m\.$", "", string).strip()
    string = regex.sub(r"(\d)\s*t$", r"\1", string).strip()

    # remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$", "")

    # string = string.replace("\\text", "")
    string = string.replace("x\\in", "")

    # remove percentage
    string = string.replace("\\%", "%")
    string = string.replace("\%", "%")
    # string = string.replace("%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    # cdot
    string = string.replace("\\cdot", "")

    # inf
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")

    # and 
    # string = string.replace("and", "")
    string = string.replace("\\mathbf", "")
    string = string.replace("\\mathrm", "")

    # use regex to remove \mbox{...}
    string = re.sub(r"\\mbox{.*?}", "", string)

    # quote
    string.replace("'", "")
    string.replace("\"", "")
    
    # i, j
    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    # replace a.000b where b is not number or b is end, with ab, use regex
    string = re.sub(r"(\d+)\.0+([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0+$", r"\1", string)

    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    string = _fix_sqrt(string)
    string = _fix_tan(string)
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    string = regex.sub(r"(\\|,|\.)+$", "", string)

    return string

def extract_boxed_answers(text):
    answers = []
    for piece in text.split('boxed{')[1:]:
        n = 0
        for i in range(len(piece)):
            if piece[i] == '{':
                n += 1
            elif piece[i] == '}':
                n -= 1
                if n < 0:
                    if i + 1 < len(piece) and piece[i + 1] == '%':
                        answers.append(piece[: i + 1])
                    else:
                        answers.append(piece[:i])
                    break
    return answers


def parse_fraction(input_str):
    """
    Parses LaTeX-like fraction strings (including negatives) and evaluates them as float decimals.
    If execution exceeds the timeout limit, it raises an error.
    """
    cleaned_str = input_str.replace("\\dfrac", "").replace("\\frac", "")
    
    is_negative = cleaned_str.startswith("-")
    
    # Strip the negative sign for easier parsing
    if is_negative:
        cleaned_str = cleaned_str[1:]
    
    cleaned_str_lst = cleaned_str.strip("{}").split("}{")
    if len(cleaned_str_lst) != 2:
        return None
    else:
        numerator = cleaned_str_lst[0]
        denominator = cleaned_str_lst[1]
    if re.fullmatch(r"-?\d+", numerator):
        numerator = int(numerator)
    else:
        return None
    if re.fullmatch(r"-?\d+", denominator):
        denominator = int(denominator)
        if denominator == 0:
            return None
    else:
        return None
    
    result = numerator / denominator
    
    if is_negative:
        result = -result
    
    return str(float(result))


def handle_minor_cases(ans, in_the_form=False, separated_by_commas=False):
    # get rid of "~~~ is"
    ans = ans.split('is')[-1].strip()

    # remove \( answer \) -> check qwen 3b
    ans = re.sub(r'\\\((.*?)\\\)', r'\1', ans)

    # remove ** answer ** -> check gemma
    ans = re.sub(r'\*\*(.*?)\*\*', r'\1', ans)

    # remove ** in front and back -> check gemma
    ans = ans.lstrip("**")
    ans = ans.rstrip("**")
    
    # Extract the last single alphabet inside parentheses (A), (B), etc.
    matches = re.findall(r'\(([A-Za-z])\)', ans)
    if matches:
        ans = matches[-1]

    # Split words and check if all except the last are alphabetic (e.g. "degree by 42")
    words = ans.split()
    if len(words) > 1 and all(re.match(r"^[A-Za-z']+$", word) for word in words[:-1]):
        ans = words[-1]
            
    # Get rid of units: if last element is alphabetic and second-last is not
    list_temp = ans.split()
    if len(list_temp) > 1:
        if list_temp[-1].isalpha():
            if re.search(r'\d', list_temp[-2]):
                ans = " ".join(list_temp[:-1])  # Remove the last word (unit)
            elif len(list_temp) > 2 and re.search(r'\d', list_temp[-3]):
                ans = " ".join(list_temp[:-2])  # Remove last two words (unit) (e.g. "square meters")

    # get rid of x=, y=
    if not in_the_form:
        ans = ans.split('=')[-1].strip()

    # get rid of , except in interval notation
    if not separated_by_commas and not re.search(r'\(\d+,\d+\)', ans):
        pattern = r'-?\d{1,3}(?:,\d{3})*(?:\.\d+)?'
        ans = re.sub(pattern, lambda x: x.group().replace(",", ""), ans)

    # get rid of %
    ans = ans.rstrip("%")

    # get rid of °
    ans = ans.rstrip("°")

    # split " \\in "
    ans = ans.split(" \\in ")[-1].strip()

    # Remove trailing zeros from decimals
    if re.match(r'^-?\d+\.\d+$', ans):
        ans = ans.rstrip('0').rstrip('.')
    
    return ans


def extract_answer(pred_str, in_the_form=False, separated_by_commas=False, exhaust=False):
    pred = []

    if 'boxed' in pred_str:
        pred = extract_boxed_answers(pred_str)
    elif ('answer is' in pred_str):
        temp = pred_str.split('answer is')[-1].strip()
        # remove the rest of the words after the answer
        temp = temp.split('.\n')[0].strip()
        temp = temp.split('. ')[0].strip()
        if " or " in temp:
            pred = temp.split(" or ")
        else:
            pred = [temp]
        
    else: 
        # use the last word that has number
        words = pred_str.split()
        for word in reversed(words):
            if re.search(r'\d', word):
                pred.append(word)
                break

        if pred == []:
            pred.append(pred_str)
       

    # multiple line
    _pred = []
    for ans in pred:
        ans = ans.strip()
        ans = ans.lstrip(":")
        ans = ans.rstrip(".")
        ans = ans.rstrip("/")
        # remove $ signs
        ans = ans.replace("\\$", "")
        ans = ans.replace("$", "")

        # handle minor cases
        ans = handle_minor_cases(ans, in_the_form, separated_by_commas)
        ans = strip_string(ans)
        _pred.append(ans)

    if exhaust:
        return _pred
    else:
        return _pred[-1] if _pred else ""


def extract_math_answer(question, reasoning):
    answer = []
    # handle cases where desired answer is multiple separated by commas => return list of lists
    in_the_form = False
    if 'in the form' in question:
        in_the_form = True
    if 'separated by commas' in question:
        for ans in extract_answer(reasoning, in_the_form=in_the_form, separated_by_commas=True, exhaust=True):
            ans = ans.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
            temp = [a.strip() for a in ans.split(",")]
            answer.append(temp)
            # handle fractions
            num_temp = []
            frac_flag = False
            for a in temp:
                if 'frac' in a:
                    frac_a = parse_fraction(a)
                    if frac_a is not None:
                        num_temp.append(frac_a)
                        frac_flag = True
                    else:
                        num_temp.append(a)
                else:
                    num_temp.append(a)
            if frac_flag:
                answer.append(num_temp)
        return answer
    # handle cases where desired answer is one (but could be multiple) => return list of strings
    else:
        for ans in extract_answer(reasoning, in_the_form=in_the_form, separated_by_commas=False, exhaust=True):
            if regex.search(r"\\text\{\s*and\s*\}", ans):
                temp = [a.strip() for a in regex.sub(r"\\text\{\s*and\s*\}", "[SEP]", ans).split("[SEP]")]
                for a in temp:
                    if 'frac' in a:
                        frac_a = parse_fraction(a)
                        if frac_a is not None:
                            temp.append(frac_a)
                answer.extend(temp)
            else:
                answer.append(ans.strip())
                if 'frac' in ans.strip():
                    frac_a = parse_fraction(ans.strip())
                    if frac_a is not None:
                        answer.append(frac_a)
        return list(set(answer))


def compare_answers(question, parsed_label, parsed_answer):
    # Convert to lists if they're string representations of lists
    if isinstance(parsed_label, str):
        parsed_label = ast.literal_eval(parsed_label)
    if isinstance(parsed_answer, str):
        parsed_answer = ast.literal_eval(parsed_answer)
    
    if 'separated by commas' in question:
        # Convert each sublist to a frozen set for hashability
        answer_sets = {frozenset(str(x) for x in ans) for ans in parsed_answer}
        label_sets = {frozenset(str(x) for x in lab) for lab in parsed_label}
        # Check for any intersection between answer_sets and label_sets
        return bool(answer_sets & label_sets)
    else:
        # Convert lists to sets for efficient membership testing
        return bool(set(str(x) for x in parsed_answer) & set(str(x) for x in parsed_label))
    

## MBPP Dataset Evaluation Functions ##

def tensor_to_python(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist() if obj.dim() > 0 else obj.item()
    elif isinstance(obj, dict):
        return {k: tensor_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_python(v) for v in obj]
    else:
        return obj


def _run_code_and_tests(code, tests):
    namespace = {}
    try:
        exec(code, namespace)
        for test in tests:
            stmt = test.strip()
            if stmt.startswith('assert '):
                stmt = stmt[len('assert '):]
            if not eval(stmt, namespace):
                return False
        return True
    except Exception:
        return False


def evaluate_task(task, tests, timeout=3):
    code_to_execute = ""
    code_pattern = r'```python(.*?)```'
    code_blocks = re.findall(code_pattern, task['generation'], re.DOTALL)
    code_to_execute = code_blocks[0] if code_blocks else task['generation']
    # tests = task['tests']
    with multiprocessing.Pool(processes=1) as pool:
        async_res = pool.apply_async(_run_code_and_tests, (code_to_execute, tests))
        try:
            passed = async_res.get(timeout=timeout)
        except multiprocessing.TimeoutError:
            print(f"⏱ Task {task.get('task_id')} timed out after {timeout}s")
            passed = False
    return passed