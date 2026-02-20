# Describe your chosen model — Write a short paragraph (3–5 sentences) as a comment at the top of the file explaining how the model works and why you chose it.
# I chose support vector machines(SVM) for this comparison. 
# SVM is an algorithm that finds the optimal hyperplane to separate classes by maximizing the margin between them. 
# It thrives with high-dimensional data like the breast cancer dataset and handles both linearly and non-linearly separable data through kernel tricks. 
# SVM is computationally efficient and provides good generalization, making it ideal for comparing against a from-scratch binary classifier. 
# Due to SVM's robust nature against cases where the number of dimensions is high, it provides a reliable benchmark.

# Implement it — Train your chosen sklearn model on the same breast cancer dataset (use load_data() from binary_classification.py).
from sklearn.svm import SVC
from binary_classification import load_data, train, predict, accuracy

#Load data
X_train, X_test, y_train, y_test, _ = load_data()

#Train from-scratch model
w, b, _ = train(X_train, y_train, verbose=False)
scratch_preds = predict(X_test, w, b)
scratch_acc = accuracy(y_test, scratch_preds)

#Convert tensors
X_train_np = X_train.numpy()
X_test_np = X_test.numpy()
y_train_np = y_train.numpy()
y_test_np = y_test.numpy()

#Train SVM
svm_model = SVC(kernel="rbf", random_state=42)
svm_model.fit(X_train_np, y_train_np)

#Evaluate SVM
svm_accuracy = svm_model.score(X_test_np, y_test_np)


#Compare — Print the test accuracy of both your from-scratch model and the sklearn model, and write a brief comment (2–3 sentences) discussing which performed better and why that might be.
print(f"Scratch Model Accuracy: {scratch_acc:.4f}")
print(f"SVM Accuracy: {svm_accuracy:.4f}")
# The SVM model performs slightly worse than the from-scratch model.
# This is likely due to the small and clean dataset we used, and the from-scratch model is basically linear regression, which is a better fit for this dataset since it's almost linearly separable.