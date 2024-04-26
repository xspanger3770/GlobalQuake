package globalquake.core.exception.action;

import javax.swing.AbstractAction;
import java.awt.event.ActionEvent;
import java.awt.event.KeyEvent;

/**
 * Action responsible for terminating the application.
 */
public final class TerminateAction extends AbstractAction {

    public TerminateAction() {
        super("Terminate");
        putValue(SHORT_DESCRIPTION, "Terminates the application");
        putValue(MNEMONIC_KEY, KeyEvent.VK_T);
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        System.exit(0);
    }
}
