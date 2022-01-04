"""Игра: Угадай число за меньшее количество попыток"""

import numpy as np

def game_core(number: int = 1) -> int:
    
    attempt_counter = 0
    prdict_number_min = 1
    prdict_number_max = 101
    
    while True:
        attempt_counter += 1
        
        predict_number = (prdict_number_max + prdict_number_min) // 2
        
        if number > predict_number:
            prdict_number_min = predict_number
        
        elif number < predict_number:
            prdict_number_max = predict_number
            
        else:
            break
        
    return attempt_counter

print(f"Количество попыток: {game_core()}")


def score_game(game_core) -> int:
    """За какое количество попыток в среднем за 1000 подходов угадывает наш алгоритм

    Args:
        random_predict ([type]): функция угадывания

    Returns:
        int: среднее количество попыток
    """

    count_ls = [] # список для сохранения количества попыток
    np.random.seed(1) # фиксируем сид для вопроизводимости
    random_array = np.random.randint(1, 101, size=(1000)) # загадали список чисел

    for number in random_array:
        count_ls.append(game_core(number))

    score = int(np.mean(count_ls)) # находим среднее количество попыток

    print(f'Ваш алгоритм угадывает число в среднем за: {score} попыток')
    return(score)

# RUN
score_game(game_core)
    

        
        