package globalquake.exception.action;

import java.awt.Window;
import java.awt.event.ActionEvent;
import java.awt.event.KeyEvent;

import javax.swing.AbstractAction;

/**
 * Action responsible for terminating the application.
 */
public final class CloseAction extends AbstractAction {

	private final Window frame;

	public CloseAction(Window frame) {
        super("Close");
        this.frame = frame;
        putValue(SHORT_DESCRIPTION, "Save everything and close normally");
        putValue(MNEMONIC_KEY, KeyEvent.VK_C);
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        if(frame != null) {
            frame.dispose();
        }
        System.exit(0);
    }
}
