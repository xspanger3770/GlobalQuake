package globalquake.ui.action.source;

import com.opencsv.CSVWriter;
import globalquake.core.database.StationDatabaseManager;
import globalquake.core.database.StationSource;
import globalquake.core.exception.RuntimeApplicationException;
import org.tinylog.Logger;

import javax.swing.*;
import javax.swing.filechooser.FileNameExtensionFilter;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

public class ExportStationSourcesAction extends AbstractAction {

    private final StationDatabaseManager databaseManager;
    private final Window parent;

    public ExportStationSourcesAction(Window parent, StationDatabaseManager databaseManager) {
        super("Export");
        this.databaseManager = databaseManager;
        this.parent = parent;

        putValue(SHORT_DESCRIPTION, "Export Seedlink Networks");
    }

    @Override
    public void actionPerformed(ActionEvent actionEvent) {
        JFileChooser chooser = new JFileChooser();
        FileNameExtensionFilter filter = new FileNameExtensionFilter(
                "CSV Files", "csv");
        chooser.setFileFilter(filter);
        chooser.setApproveButtonText("Ok");
        chooser.setDialogTitle("Export as CSV");
        chooser.setDragEnabled(false);

        int returnVal = chooser.showSaveDialog(parent);
        if (returnVal == JFileChooser.APPROVE_OPTION) {
            File selectedFile = chooser.getSelectedFile();
            // Check if the selected file has ".csv" suffix, if not, append it
            String filePath = selectedFile.getAbsolutePath();
            if (!filePath.toLowerCase().endsWith(".csv")) {
                selectedFile = new File(filePath + ".csv");
            }
            // Check if the file already exists
            if (selectedFile.exists()) {
                int result = JOptionPane.showConfirmDialog(parent, "The file already exists. Do you want to overwrite it?",
                        "File Exists", JOptionPane.YES_NO_OPTION);
                if (result != JOptionPane.YES_OPTION) {
                    // User doesn't want to overwrite the file
                    return;
                }
            }
            exportTo(selectedFile);
            JOptionPane.showMessageDialog(parent, "CSV file exported successfully!");
        }
    }

    private void exportTo(File selectedFile) {
        try (CSVWriter writer = new CSVWriter(new FileWriter(selectedFile))) {
            // Writing data to CSV file
            writer.writeAll(createData());
            Logger.info("CSV file exported successfully!");
        } catch (IOException e) {
            throw new RuntimeApplicationException("CSV export failed! %s".formatted(e.getMessage()));
        }
    }

    private java.util.List<String[]> createData() {
        java.util.List<String[]> result = new ArrayList<>();
        result.add(new String[]{"Name", "URL"});
        for (StationSource stationSource : databaseManager.getStationDatabase().getStationSources()) {
            result.add(new String[]{
                    stationSource.getName().replace('"', ' '),
                    stationSource.getUrl().replace('"', ' '),
            });
        }
        return result;
    }

}
