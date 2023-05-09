from matplotlib import pyplot as plt


def make_video_with_caption(videos, captions, n_samples, title='Video', timestep=None, change_color=None):
    n_videos = len(videos)

    fig, axes = plt.subplots(nrows=1, ncols=n_videos, figsize=(n_videos*1.4, n_videos*1.4))

    fig.title(title)

    if timestep is not None:

        fig.subtitle(f't={timestep}', )

    for row in axes:
        for ax in row:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    for ax, col in zip(axes[0], captions):
        ax.set_title(col)


        


