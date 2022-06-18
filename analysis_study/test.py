from sklearn.metrics import matthews_corrcoef


x = [+1, -1, +1, +1] # список значений признака х
y = [+1, +1, +1, -1] # список значений признака y

print(matthews_corrcoef(x, y)) # рассчитаем коэффициент корреляции Мэтьюса
