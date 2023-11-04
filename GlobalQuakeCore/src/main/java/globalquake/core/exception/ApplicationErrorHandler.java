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

	private Window parent;

	private int errorCount = 0;


	public ApplicationErrorHandler(Window parent) {
		this.parent = parent;
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

	public synchronized void handleWarning(Throwable e) {
		showWarning(e.getMessage());
	}

	private void showWarning(String message) {
		JOptionPane.showMessageDialog(parent, message, "Warning", JOptionPane.WARNING_MESSAGE);
	}

	private void showDetailedError(Throwable e) {
		errorCount++;
		if (errorCount == 2) {
			System.exit(0);
		}
		final Object[] options = getOptionsForDialog(true);
		JOptionPane.showOptionDialog(parent, createDetailedPane(e), "Fatal Error", JOptionPane.DEFAULT_OPTION,
				JOptionPane.ERROR_MESSAGE, null, options, null);
		errorCount = 0;
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
		final Object[] options = getOptionsForDialog(isFatal);

		JOptionPane.showOptionDialog(parent, message, title, JOptionPane.DEFAULT_OPTION, JOptionPane.ERROR_MESSAGE, null,
				options, null);
	}

	private Component[] getOptionsForDialog(boolean isFatal) {
		if (!isFatal) {
			return null; // use default
		}

		return new Component[] { new JButton(new TerminateAction()), new JButton(new OpenURLAction("https://github.com/xspanger3770/GlobalQuake/issues", "Open issue on GitHub")),
				new JButton(new IgnoreAction()) };
	}
}
