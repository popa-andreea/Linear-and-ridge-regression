# Linear-and-ridge-regression

Exercitiul 1
-
functia normalize(train_data, test_data):
-
- definim si antrenam modelul pe datele de antrenare date ca parametru (train_data);
- scalam datele de antrenare;
- verficam daca datele de testare date ca parametru (test_data) sunt de tipul None;
- daca da, returnam doar datele de antrenare;
- altfel, scalam si datele de testare.


Exercitiul 2
-
- incarcam datele de antrenare (training_data) si etichetele (prices) si le impartim pe ambele in trei vectori (numpy arrays) de lungimi egale.

functia linear_regression(train_data, train_labels, test_data, test_labels):
- 
- normalizam datele de antrenare si datele de testare;
- definim si antrenam modelul de regresie liniara pe datele de antrenare;
- prezicem etichetele datelor de testare;
- calculam mean absolute error (mae), care reprezinta valoarea medie a diferantei absolute dintre valorile estimate si cele reale;
- si mean squared error (mse), care reprezinta valoarea medie a difentei la patrat dintre valorile prezise si valorile reale.

functia linear_mae_mse():
-
- salvam cele trei mae si mse rezultate in urma apelarii functiei linear_regression pentru datele de antrenare si etichetele impartite in trei;
- calculam valoare medie a celor 3 mae si mse.

Exercitiul 3
-
functia linear_regression(train_data, train_labels, test_data, test_labels, alpha):
- 
- normalizam datele de antrenare si datele de testare;
- definim si antrenam modelul de regresie ridge cu paramentrul alpha pe datele de antrenare;
- prezicem etichetele datelor de testare;
- calculam mean absolute error (mae), care reprezinta valoarea medie a diferantei absolute dintre valorile estimate si cele reale;
- si mean squared error (mse), care reprezinta valoarea medie a difentei la patrat dintre valorile prezise si valorile reale.

functia ridge_mae_mse():
- 
- pentru fiecare alpha din intervalul [1, 10, 100, 1000]:
- salvam cele trei mae si mse rezultate in urma apelarii functiei ridge_regression pentru datele de antrenare si etichetele impartite in trei;
- calculam valoare medie a celor 3 mae si a celor 3 mse;
- in cazul primului element din interval, initializam variabila best;
- verficam daca pentru parametrul alpha performanta este mai buna decat pentru parametrul best;
- daca da, actualizam variabila best;
- variabila best reprezinta parametrul cu cea mai buna performanta.

Exercitiul 4
-
functia ridge_(train_data, train_labels):
- folosim functia definita anterior pentru a afla paramentrul cu cea mai buna performanta;
- normalizam datele de antrenare;
- definim si antrenam modelul de regresie ridge pe datele de antrenare;
- calculam si afisam coeficienti si bias-ul;
- afisam cel mai semnificativ si cel mai putin semnificativ atribut cu ajutorul functiilor np.argmax si np.argmin, care returneaza indicele valorii maxime/minime dintr-un array;
- afisam al doilea cel mai semnificativ atribut cu ajutorul functiei np.argpartition care returneaza un array de indici a elementelor partitionate din care luam doar penultimul element.









