import av
container = av.open(r"hand.mp4")

for frame in container.decode(video=0):
    frame.to_image().save('videoframe-%04d.png' % frame.index)