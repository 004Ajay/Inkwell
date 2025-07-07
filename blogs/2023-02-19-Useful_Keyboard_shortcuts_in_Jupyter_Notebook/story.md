---
slug: jupyter-notebook-keyboard-shortcuts
title: "Useful Keyboard shortcuts in Jupyter Notebook"
authors: [ajay]
tags: [jupyter-notebook]
---

Read on [Medium](https://medium.com/@ajaytshaju/useful-keyboard-shortcuts-in-jupyter-notebook-3d488c1e5d29)

How to work faster in jupyter notebooks — the answer is _Keyboard shortcuts_.

Keyboard shortcuts are a great way to do digital work faster, when it comes to programming it’s more relevant. With this article, I am sharing some of my daily keyboard shortcuts while using jupyter notebook.

<!-- truncate -->

Jupyter Notebook is a powerful python code editor which can be installed on your local machine and works in the browser. I have been coding with jupyter notebook for about two years now and using a keyboard shortcut from the first week itself (i like to do things without touching the mouse, it seems more professional 😄)

You can always read more about jupyter notebook on [Wikipedia](https://en.wikipedia.org/wiki/Project_Jupyter)

![The interface of a new jupyter notebook](The%20interface%20of%20a%20new%20jupyter%20notebook.webp)
The interface of a new jupyter notebook

_Note: This article may help windows & Linux users directly, for MAC users, there will be some changes in keys — **ctrl → command, Alt → Option**_

_When a jupyter notebook starts, we could see whole keyboard shortcuts by pressing `esc` & `h`_

The first thing most of us may do on jupyter notebook is to change the title but as of now, there is no default keyboard shortcut for title renaming, but I have set a custom keyboard shortcut for renaming the title— which will be explained at the next section of this article

Now, let’s jump on to excellent jupyter notebook keyboard shortcuts

## Keyboard Shortcuts

### 1. Escape key to exit the current cell

When the notebook starts, the first cell will be active(cell edit mode) so to change the mode to command mode — simply press `esc` key

_we could see this change by the change in color of this cell outline_

![An active cell](An%20active%20cell.webp)
An active cell

![An inactive cell i.e command mode](An%20inactive%20cell%20ie%20command%20mode.webp)
An inactive cell i.e command mode

_These active and command modes are important, most commands given in this article work on command mode_

### 2. Letter ‘A’ to add multiple cells above (command mode)

We need more cells to work on, we could add more cells by clicking the letter **A** multiple times. We could add cells below by clicking **B**, but we use A initially to make the cell selection line(cell outline) stay at the top cell. We could move up & down through cells using the **UP & DOWN** arrow keys, and **ENTER** key to enter the selected cell.

![A notebook with multiple cells](A%20notebook%20with%20multiple%20cells.webp)
A notebook with multiple cells

### 3. Autofill code using `tab` key (active mode)

After entering a cell and while writing the code, it is very helpful if the code editor has a code completion like [PyCharm](https://www.jetbrains.com/pycharm/) & [VS Code](https://code.visualstudio.com/). Jupyter notebook also has this functionality in `tab` key. You just need to press the tab key wherever you need to complete the code, this could be a variable name, a function name, etc…

![Code completion](Code%20completion.webp)
Code completion

### 4. Run the current cell and select the next cell using `Shift+Enter` (active mode)

After entering a cell and writing some code, we need to run it right? — then press `shift+enter` to run the cell & go to the next cell.

![A cell with an output](A%20cell%20with%20an%20output.webp)
A cell with an output

### 5. To see function documentation & examples — `shift+tab` (active mode)

If we don’t know what are the arguments & return of a function, we could find them by pressing `shift+tab`

_Note: you need to run the import statement to get the documentation of a function_

![Function documentation & examples](Function%20documentation%20&%20examples.webp)
Function documentation & examples

### 6. Showing Line numbers — `shift+L` (command mode)

By default, the line numbering is off in jupyter notebook but it is very helpful to see the line numbers, especially when our code has some errors. You can turn on the line numbers by clicking `shift+L` in command mode.

![A cell with line numbers](A%20cell%20with%20line%20numbers.webp)
A cell with line numbers

### 7. Find and replace using the ‘F’ key (command mode)

Changing a variable name that is written in different places is a tedious task, we could do it simply using `find and replace` just like many other applications. In jupyter notebook, you just need to press the letter **F** in command mode(blue outline around the cell)

Enter the word to find in the `Find` box and word to replace in the `Replace` box

![Find and replace box](Find%20and%20replace%20box.webp)
Find and replace box

This ↓ is an example usage of find and replace. The words enclosed in the red box will be replaced with words in the green box

![Find and replace sample](Find%20and%20replace%20sample.webp)
Find and replace sample

### 8. M for Markdown and Y for code (command mode)

If you finished writing the code and you are ready to explain or send it to your friend or professor, then it would be a good practice to add some Markdown or LaTeX commands in your notebook to explain the idea more clearly.

_Markdown is a simple markup language to write formatted text. It will take only 1 or 2 hours to study the markdown_

_LaTeX is a software used for document preparation, especially research papers — extra info 😉_

For this, just make a new cell above(A) or below(B) and just press the letter M to convert the cell to a markdown cell. _The Markdown cells do not have the `In[ ]` at the cell’s starting._

![A markdown cell](A%20markdown%20cell.webp)
A markdown cell

![An executed markdown cell](An%20executed%20markdown%20cell.webp)
An executed markdown cell

To convert your cell to a normal coding cell, just press the letter Y.

![A code cell](A%20code%20cell.webp)
A code cell

### 9. Delete cells using `dd` (command mode)

It is better to delete the unused cells for the final draft of your notebook, just press the letter `d` twice to delete the current cell. You can undo this action by pressing `Z`

* User-Defined Keyboard Shortcuts — I have defined 2 shortcuts in my notebook, to make my workflow faster.

---

How to define your own Keyboard Shortcuts in jupyter notebook?

`Help → Edit Keyboard Shortcuts`

![Edit Keyboard Shortcuts](Edit%20Keyboard%20Shortcuts.webp)

## Edit Keyboard Shortcuts

![The new keyboard shortcut for running all cells](The%20new%20keyboard%20shortcut%20for%20running%20all%20cells.webp)
The new keyboard shortcut for running all cells

Here, I am adding `Alt+Ctrl+R` a key bind for `run all cells`. Hit enter to commit the new shortcut and click `OK` button on the right bottom.

_Make sure, your new shortcut does not exist in the notebook & if your new shortcut contains multiple keys, separate them using hyphen `-`_

![All cells executed](All%20cells%20executed.webp)

All cells executed

### 1. Renaming the notebook (command mode)

I have added a shortcut for renaming the notebook by pressing the letter N.

![Custom shortcut for renaming notebook](Custom%20shortcut%20for%20renaming%20notebook.webp)

Custom shortcut for renaming notebook

![When the letter N is pressed](When%20the%20letter%20N%20is%20pressed.webp)
When the letter N is pressed

### 2. Move cells up and down (command mode)

When we have some relevant codes in a cell, it is more faster to move the cells up or down than cut and paste.

I have added a shortcut to move cells using `Alt+Up arrow` for moving cells up and `Alt+Down arrow` for moving cells down.

![Before moving the cell up](Before%20moving%20the%20cell%20up.webp)
Before moving the cell up

![After moving the cell up](After%20moving%20the%20cell%20up.webp)  
After moving the cell up

---

**Reached till here?**

Some extra jupyter notebook tips and tricks for you

## Extra Tips

### 1. Re-run your notebook for perfect results

When you finish writing the code and you got the result, don't stop the process but do, `Kernel → Restart & Clear Output` to restart the kernel and clear all the outputs from your notebook. Then re-run all the cells by `Cell → Run All`.

_By doing this, the notebook will clear all the saved variables from its environment and redo all the processes again, it may be sometimes computationally intensive but give you the perfect results._

### 2. Presenting the notebook

Presenting the notebook separately in a tab of your browser will make you stand out. To do this, activate the presentation mode `view → cell toolbar → slideshow`

It will open up a drop-down at the right side of each cell

![Slide modes](Slide%20modes.webp)
Slide modes

* `Slide` - the cell will be a new slide

* `Sub-Slide` - the cell will be shown in the current slide as a replacement for previous content. It will be available in arrow-down navigation

* `Fragment` - the cell will appear in the current slide, and it will append to the previous content. It will be available in arrow-down and arrow-right navigation

* `Skip` - the content will not be displayed in the presentation

* `Notes` - notes for slide, the cell content is not displayed in the presentation

_These explanations are taken from [here](https://mljar.com/blog/jupyter-notebook-presentation/)_

After setting the required modes, we can export the .ipynb file to the presentation file using

```bash
jupyter nbconvert <notebook-name>.ipynb --to slides --post serve
```
* `--to slides` to convert the .ipynb file to a presentation file .html

* `--post serve` to host the presentation file on the local HTTP server

Now you could see something like this ↓

![Notebook Presentation](Notebook%20Presentation.gif)

Notebook Presentation

That's all, thank you for reading the article, hope it helps to make your workflow faster.