package globalquake.core.exception;

import globalquake.core.action.OpenURLAction;
import globalquake.core.exception.action.IgnoreAction;
import globalquake.core.exception.action.TerminateAction;
import org.tinylog.Logger;

import javax.swing.*;
import java.awt.*;
import java.io.PrintWriter;
import java.io.StringWriter;

public class ApplicationErrorHandler implements Thread.UncaughtExceptionHandler {

    private final boolean headless;
    private Window parent;

    private int errorCount = 0;


    public ApplicationErrorHandler(Window parent, boolean headless) {
        this.parent = parent;
        this.headless = headless;
    }

    public void setParent(Window parent) {
        this.parent = parent;
    }

    @Override
    public void uncaughtException(Thread t, Throwable e) {
        Logger.error("An uncaught exception has occurred in thread {} : {}", t.getName(), e.getMessage());
        Logger.error(e);
        handleException(e);
    }

    public synchronized void handleException(Throwable e) {
        Logger.error(e);

        if (headless) {
            return;
        }

        if (e instanceof OutOfMemoryError) {
            showOOMError(e);
            return;
        }

        if (!(e instanceof RuntimeApplicationException)) {
            showDetailedError(e);
            return;
        }

        if (e instanceof FatalError ex) {
            showGeneralError(ex.getUserMessage(), true);
        } else {
            ApplicationException ex = (ApplicationException) e;
            showGeneralError(ex.getUserMessage(), false);
        }
    }

    private void showOOMError(Throwable e) {
        Logger.error(e);
        final Object[] options = getOptionsForDialog(true, false);
        JOptionPane.showOptionDialog(parent, createOOMPanel(), "Out of memory!", JOptionPane.DEFAULT_OPTION,
                JOptionPane.ERROR_MESSAGE, null, options, null);
    }

    public synchronized void handleWarning(Throwable e) {
        Logger.warn(e);

        if (headless) {
            return;
        }

        showWarning(e.getMessage());
    }

    private void showWarning(String message) {
        JOptionPane.showMessageDialog(parent, message, "Warning", JOptionPane.WARNING_MESSAGE);
    }


    public void info(String s) {
        if (headless) {
            Logger.info(s);
        } else {
            JOptionPane.showMessageDialog(parent, s, "Info", JOptionPane.INFORMATION_MESSAGE);
        }
    }

    private void showDetailedError(Throwable e) {
        errorCount++;
        if (errorCount == 2) {
            System.exit(0);
        }
        final Object[] options = getOptionsForDialog(true, true);
        JOptionPane.showOptionDialog(parent, createDetailedPane(e), "Fatal Error", JOptionPane.DEFAULT_OPTION,
                JOptionPane.ERROR_MESSAGE, null, options, null);
        errorCount = 0;
    }

    private Component createOOMPanel() {
        JPanel panel = new JPanel(new BorderLayout());

        JPanel labelsPanel = new JPanel(new GridLayout(2, 1));

        labelsPanel.add(new JLabel("GlobalQuake has run out of memory!"));
        labelsPanel.add(new JLabel("Please select less stations or connect to server."));

        panel.add(labelsPanel, BorderLayout.NORTH);

        return panel;
    }

    private Component createDetailedPane(Throwable e) {
        JPanel panel = new JPanel(new BorderLayout());

        JPanel labelsPanel = new JPanel(new GridLayout(2, 1));

        labelsPanel.add(new JLabel("Oops! Something has gone terribly wrong inside GlobalQuake."));
        labelsPanel.add(new JLabel("Please send the following text to the developers so that they can fix it ASAP:"));

        panel.add(labelsPanel, BorderLayout.NORTH);

        JTextArea textArea = new JTextArea(16, 60);
        textArea.setEditable(false);
        StringWriter stackTraceWriter = new StringWriter();
        e.printStackTrace(new PrintWriter(stackTraceWriter));
        textArea.append(stackTraceWriter.toString());

        panel.add(new JScrollPane(textArea), BorderLayout.CENTER);
        return panel;
    }

    private void showGeneralError(String message, boolean isFatal) {
        final String title = isFatal ? "Fatal Error" : "Application Error";
        final Object[] options = getOptionsForDialog(isFatal, true);

        JOptionPane.showOptionDialog(parent, message, title, JOptionPane.DEFAULT_OPTION, JOptionPane.ERROR_MESSAGE, null,
                options, null);
    }

    private Component[] getOptionsForDialog(boolean isFatal, boolean github) {
        if (!isFatal) {
            return null; // use default
        }

        if (github) {

            return new Component[]{new JButton(new TerminateAction()), new JButton(new OpenURLAction("https://github.com/xspanger3770/GlobalQuake/issues", "Open issue on GitHub")),
                    new JButton(new IgnoreAction())};
        } else {
            return new Component[]{new JButton(new TerminateAction()), new JButton(new IgnoreAction())};
        }
    }

}
