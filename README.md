# DRA-GAN
🎨 Fake-Vision : My DRAGAN Adventure! 🚀
Hey There! Welcome to My Project! 😄
Hi, I’m so excited to share my cool project with you! I built a Fake-Vision using something called DRAGAN (a fancy type of Generative Adversarial Network, or GAN for short). Basically, I taught my computer to create fake human faces that look super real—well, almost! I trained it using 50,000 pictures of celebrity faces from the CelebA dataset, and watched it get better and better over time. Let’s dive into the fun details!
What’s This All About? 🤔
Imagine two friends playing a game: one friend (the Generator) tries to draw fake faces, and the other friend (the Discriminator) guesses if the face is real or fake. At first, the drawings are pretty bad, but as they keep playing, the Generator gets really good at making faces that can fool the Discriminator! That’s what a GAN does, and DRAGAN is a special version that makes the game even more stable so the faces look better.
How Did I Do It? 🛠️
Here’s the simple breakdown of my project:
Step 1: Gathering the Faces 📸
I used 50,000 pictures from the CelebA dataset (a big collection of celebrity faces). I resized all the pictures to 64x64 pixels so my computer could handle them easily.
Step 2: Building the Face Maker (Generator) and Face Checker (Discriminator) 🖥️

Generator: This is the artist! It starts with random noise (like scribbles) and turns it into a face using lots of math magic (neural network layers).
Discriminator: This is the judge! It looks at a picture and says, “Hmm, is this a real face or a fake one?”

Step 3: Training the Team 🏋️
I let my Generator and Discriminator play their game for 25 rounds (called epochs). Each round, they looked at 64 pictures at a time (batch size = 64). The Generator tried to get better at fooling the Discriminator, and the Discriminator tried to get better at spotting fakes. I used a tool called PyTorch to make this happen, and it took a while, but it was worth it!

Random Noise → Generator Makes a Fake Face → Discriminator Guesses: Real or Fake? → Both Learn and Improve → Repeat!

Results Over Time 📈
Let’s see how my Fake-Vision improved over the 25 rounds of training! I saved a grid of 100 faces after each round, and you can see the progress below.

Round 1 (Starting Point): Just a mess of colors—no faces yet!

Round 5: Starting to see some shapes, but still very blurry.

Round 10: Hey, I can see eyes and mouths now—progress!

Round 15: Looking more like faces, but still a bit weird.

Round 20: Getting closer to real faces, but some smudges remain.

Round 24 (Almost Done): Wow, these faces look pretty good now!

Try It Yourself! 🕹️
Want to make your own fake faces? Here’s how:

Clone This Repo: Download my code to your computer.
Get the CelebA Dataset: You can find it here. Put the images in a folder on your computer.
Update the Code: Open dragan.py and change the dataset path in the Args class to point to your CelebA folder.
Run It: Use Python to run dragan.py, and watch the magic happen!

Try changing some numbers in the code—like the input_size (to make bigger faces) or z_dim (to change how random the faces are)—and see what happens!
What’s Next? 🚀

Make Faces Bigger: I want to try making 128x128 faces instead of 64x64 to see if they look sharper.
Add More Variety: Sometimes the faces looked too similar, so I’ll try tricks to make them more different.
Use It for Good: Maybe I can use these fake faces to help train a face recognition app—or even help spot deepfakes!

Fun Facts & Challenges 🎯

Fun Fact: The DRAGAN trick I used helps the Generator and Discriminator play nicely together, so the faces don’t turn out too crazy!
Challenge: Sometimes the Generator got lazy and made the same face over and over (this is called “mode collapse”). I had to tweak the settings to fix it.
