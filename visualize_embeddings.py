import sys
import plotly
import numpy as np
import plotly.graph_objs as go
from MulticoreTSNE import MulticoreTSNE as TSNE
import compress_fasttext
import gensim.downloader as gloader


def append_list(sim_passwords, password):
    list_of_passwords = []
    
    for i in range(len(sim_passwords)):
        sim_passwords_list = list(sim_passwords[i])
        sim_passwords_list.append(password)
        sim_passwords_tuple = tuple(sim_passwords_list)
        list_of_passwords.append(sim_passwords_tuple)
        
    return list_of_passwords


def display_tsne_scatterplot_3D(model,
                                user_input = None,
                                words = None,
                                label = None,
                                color_map = None,
                                perplexity = 0,
                                learning_rate = 0,
                                iteration = 0,
                                topn = 5,
                                sample = 10):
    if words == None:
        if sample > 0:
            words = np.random.choice(list(model.vocab.keys()), sample)
        else:
            words = [word for word in model.vocab]
    word_vectors = np.array([model[w] for w in words])

    # For 2D, change the three_dim variable into something like two_dim like the following:
    # two_dim = TSNE(n_components = 2, random_state=0, perplexity = perplexity, learning_rate = learning_rate, n_iter = iteration).fit_transform(word_vectors)[:,:2]    
    three_dim = TSNE(n_components=3, random_state=0, perplexity=perplexity, learning_rate=learning_rate, n_iter=iteration, n_jobs=4).fit_transform(word_vectors)[:, :3]

    data = []
    count = 0
    for i in range(len(user_input)):
        # For 2D, instead of using go.Scatter3d, we need to use go.Scatter and delete the z variable. Also, instead of using
        # variable three_dim, use the variable that we have declared earlier (e.g two_dim)
        trace = go.Scatter3d(
            x=three_dim[count:count + topn, 0], 
            y=three_dim[count:count + topn, 1],  
            z=three_dim[count:count + topn, 2],
            text=words[count:count + topn],
            name=user_input[i],
            textposition="top center",
            textfont_size=20,
            mode='markers+text',
            marker={
                'size': 10,
                'opacity': 0.8,
                'color': 2
            }

        )
    
        data.append(trace)
        count = count+topn

    # For 2D, instead of using go.Scatter3d, we need to use go.Scatter and delete the z variable.  Also, instead of using
    # variable three_dim, use the variable that we have declared earlier (e.g two_dim)
    trace_input = go.Scatter3d(
        x=three_dim[count:,0], 
        y=three_dim[count:,1],  
        z=three_dim[count:,2],
        text=words[count:],
        name='input words',
        textposition="top center",
        textfont_size=20,
        mode='markers+text',
        marker={
            'size': 10,
            'opacity': 1,
            'color': 'black'
        }
    )
            
    data.append(trace_input)
    
    # Configure the layout
    layout = go.Layout(
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
        showlegend=True,
        legend=dict(
        x=1,
        y=0.5,
        font=dict(
            family="Courier New",
            size=25,
            color="black"
        )),
        font=dict(
            family=" Courier New ",
            size=15),
        autosize=False,
        width=1000,
        height=1000
    )

    plot_figure = go.Figure(data=data, layout=layout)
    plot_figure.show()


def main():
    if len(sys.argv) == 1:
        print('Insert at least a password.')
        sys.exit(-1)
    input_passwords = sys.argv[1:]
    result = []

    small_model = compress_fasttext.models.CompressedFastTextKeyedVectors.load('no_w2kp_compressed_model_minngram=2')
    
    # For each input password, find the 5 most similar ones using small_model embeddings
    for password in input_passwords:
        sim_passwords = small_model.most_similar(password, topn=5)
        sim_passwords = append_list(sim_passwords, password)
        print(sim_passwords)
        result.extend(sim_passwords)
    
    similar_password = [password[0] for password in result]
    similarity = [password[1] for password in result] 
    similar_password.extend(input_passwords)
    labels = [password[2] for password in result]
    label_dict = dict([(y, x + 1) for x, y in enumerate(set(labels))])
    color_map = [label_dict[x] for x in labels]

    display_tsne_scatterplot_3D(small_model, input_passwords, similar_password, labels, color_map, 5, 500, 10000)


if __name__ == "__main__":
    main()
