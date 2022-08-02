import preprocess as x

numb,whatisit=x.predictirl('image.jpg')

if whatisit==0:
    print("cat")
else:
    print("dog")
print(numb)
