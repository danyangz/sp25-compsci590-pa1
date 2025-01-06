FILES= auto_diff.py logistic_regression.py

handin.zip: $(FILES)
	zip handin.zip $(FILES)

clean:
	rm -f *~ handin.zip
