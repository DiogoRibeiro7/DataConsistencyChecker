"""HTML export helpers for DataConsistencyChecker."""

from __future__ import annotations

import os


class ExportMixin:
    """Mixin providing HTML export helpers."""

    def gpt_export_html(self):
        """Retained compatibility stub for the unsupported legacy exporter."""
        print("get_export_html() not supported in this version")
        return

    def export_html(self, test_id_list=None, output_file: str = "Data_consistency.html"):
        """Export patterns and exceptions to an HTML report.

        Args:
            test_id_list: Optional list of test IDs to include. If ``None`` all
                available tests are exported.
            output_file: Output path for the generated HTML file.
        """

        def print_test_header(test_id, section_name, f):
            nonlocal test_id_list
            assert section_name in ['patterns', 'exceptions']

            f.write("<br>" + "\n")
            func_name = f"func_test_div_{section_name}_{test_id}"
            div_name = f"test_div_{section_name}_{test_id}"
            script_str = """
                <script>
                function func_name() {
                    var x = document.getElementById("div_name");
                    if (x.style.display === "none") {
                        x.style.display = "block";
                    } else {
                        x.style.display = "none";
                    }
                }
                </script>
                """
            script_str = script_str.replace("func_name", func_name)
            script_str = script_str.replace("div_name", div_name)
            f.write(script_str + "\n")

            f.write(f'<p onclick="{func_name}()">{test_id}</p>')
            return div_name

        def print_column_header(col_name, f):
            f.write(f"<br>Column(s): {col_name}" + "\n")

        if test_id_list is None:
            test_id_list = self.get_test_list()

        with open(output_file, 'w') as f:
            f.write("<html>" + os.linesep)
            f.write("<head>" + os.linesep)
            f.write("</head>" + os.linesep)
            f.write("<body>" + os.linesep)
            f.write("<h1>Data Consistency Check Results</h1>" + "\n")

            f.write("<a href='#Patterns'>Patterns</a><br/>" + "\n")
            f.write("<a href='#Exceptions'>Exceptions</a><br/>" + "\n")

            f.write("<h2 id='Patterns'>Patterns</h2>" + "\n")
            for test_id in test_id_list:
                sub_patterns_test = self.patterns_df[self.patterns_df['Test ID'] == test_id]

                for columns_set in sub_patterns_test['Column(s)'].values:
                    sub_patterns = self.patterns_df[(self.patterns_df['Test ID'] == test_id) &
                                                    (self.patterns_df['Column(s)'] == columns_set)]
                    if len(sub_patterns) > 0:
                        print_test_header(test_id, "patterns", f)
                        print_column_header(columns_set, f)
                        f.write("<p>Pattern found (without exceptions)</p>")
                        f.write(sub_patterns.iloc[0]['Description of Pattern'])

            f.write("<h2 id='Exceptions'>Exceptions</h2>")
            for test_id in test_id_list:
                sub_results_summary_test = self.exceptions_summary_df[self.exceptions_summary_df['Test ID'] == test_id]
                if len(sub_results_summary_test) == 0:
                    continue
                div_name = print_test_header(test_id, 'exceptions', f)
                f.write(f'<div id="{div_name}">')
                for columns_set in sub_results_summary_test['Column(s)'].values:
                    sub_summary = self.exceptions_summary_df[(self.exceptions_summary_df['Test ID'] == test_id) &
                                                          (self.exceptions_summary_df['Column(s)'] == columns_set)]
                    if len(sub_summary) == 0:
                        continue

                    print_column_header(columns_set, f)
                    f.write(f"<br>Issue index: {sub_summary.index[0]}")
                    f.write("<br>A strong pattern, and exceptions to the pattern, were found.<br>")
                    f.write(sub_summary.iloc[0]['Description of Pattern'])
                    num_exceptions = sub_summary.iloc[0]['Number of Exceptions']
                    f.write(f"<br>Number of exceptions: {num_exceptions} "
                            f"({num_exceptions * 100.0 / self.num_rows:.4f}% of rows)")
                f.write('</div>')

            f.write("</body>")
            f.write("</html>")

