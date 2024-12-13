# Audio Synthesis Experimentation with Generative Adversarial Networks

This repo represents some work I did for a class project for my Machine and Computer Vision. I went somewhat off the beaten trail and decided rather than attempting a classification problem I would focus on a generative project. I was drawn to the idea of a 
model that could "read" and then 'play" music. Not sheet music, but spectrograms, which are a visual representation of audio, and where pixels do not have a spatial context but instead a time-frequency relationship with a different semantic meaning. I made my own GAN model,
so including the generator, discriminator, and training step/training loop logic. For testing, I used the UrbanSounds8K dataset, specifically the street_music subset, which is typified by featuring short samples of live music recorded at different urban venues, and includes
ambient noise like human chatter, laughter, cheering, singing, or passing traffic that somewhat complicate or muddy the audio. I then pre-processed and converted these audio files into "real" spectrograms using a script I'm including.

This model was to put it shortly, not a success. I'm including a report I wrote (it's an informal report, so formatting and technical writing are less to be desired) that will go into further detail about my results, but to summarize, my model could not converge. 
Through a lot of experimentation and trial and error, as well just long training hours, I did produce some spectrogram-like images, where certain characteristics from my real spectrograms could be observed and noted in my generated "fake" spectrograms. However, they still
were not perfect, and the generated audio was generally unpleasant to listen to. 

Uploading here on the off chance someone might find it online and use it as a potential resource (of what to do, but more likely what not to do), to fill up my Github a bit, and to work on and update in the future.
